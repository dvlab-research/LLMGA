import inspect
import warnings
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, Attention
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler
from llmga.diffusers.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
# from diffusers.utils import logging, randn_tensor
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.semantic_stable_diffusion import SemanticStableDiffusionPipelineOutput
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import math
from collections.abc import Iterable
import re

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: DiffusionPipeline, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: DiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(text_input)[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: DiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = getattr(pipe.tokenizer, "pad_token_id", eos)
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=text_embeddings.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=uncond_embeddings.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None

class AttentionStore():
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        bs = 2 + int(PnP) + editing_prompts
        skip = 2 if PnP else 1  # skip PnP & unconditional

        head_size = int(attn.shape[0] / self.batch_size)
        attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
        source_batch_size = int(attn.shape[1] // bs)
        self.forward(
            attn[:, skip * source_batch_size:],
            is_cross,
            place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

    def between_steps(self, store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_attention(self, step: int):
        if self.average:
            attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                         self.attention_store}
        else:
            assert (step is not None)
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(self, attention_maps, prompts, res: int,
                            from_where: List[str], is_cross: bool, select: int
                            ):
        out = [[] for x in range(self.batch_size)]
        num_pixels = res ** 2
        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out

    def __init__(self, average: bool, batch_size=1):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size


class CrossAttnProcessor:

    def __init__(self, attention_store, place_in_unet, PnP, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts
        self.PnP = PnP

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        assert (not attn.residual_connection)
        assert (attn.spatial_norm is None)
        assert (attn.group_norm is None)
        assert (hidden_states.ndim != 4)
        assert (encoder_hidden_states is not None)  # is cross

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs,
                       is_cross=True,
                       place_in_unet=self.place_in_unet,
                       editing_prompts=self.editing_prompts,
                       PnP=self.PnP)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing
class GaussianSmoothing():

    def __init__(self, device):
        kernel_size = [3, 3]
        sigma = [0.5, 0.5]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel.to(device)

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))


def load_512(image_path, size, left=0, right=0, top=0, bottom=0, device=None, dtype=None):
    def pre_process(im, size, left=0, right=0, top=0, bottom=0):
        if type(im) is str:
            image = np.array(Image.open(im).convert('RGB'))[:, :, :3]
        elif isinstance(im, Image.Image):
            image = np.array((im).convert('RGB'))[:, :, :3]
        else:
            image = im
        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h - bottom, left:w - right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((size, size)))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image

    tmps = []
    if isinstance(image_path, list):
        for item in image_path:
            tmps.append(pre_process(item, size, left, right, top, bottom))
    else:
        tmps.append(pre_process(image_path, size, left, right, top, bottom))
    image = torch.stack(tmps) / 127.5 - 1

    image = image.to(device=device, dtype=dtype)
    return image


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing

def reset_dpm(scheduler):
    if isinstance(scheduler, DPMSolverMultistepSchedulerInject):
        scheduler.model_outputs = [
                                           None,
                                       ] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0


class SemanticStableDiffusionImg2ImgPipeline_DPMSolver(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation with latent editing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    This model builds on the implementation of ['StableDiffusionPipeline']

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler,DPMSolverMultistepSchedulerInject],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) or not isinstance(scheduler, DPMSolverMultistepSchedulerInject):
            scheduler = DPMSolverMultistepSchedulerInject.from_config(scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2)
            logger.warning("This pipeline only supports DDIMScheduler and DPMSolverMultistepSchedulerInject. "
                           "The scheduler has been changed to DPMSolverMultistepSchedulerInject.")

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def progress_bar(self, iterable=None, total=None, verbose=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if not verbose:
            return iterable
        elif iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_unet(self, attention_store, PnP: bool = False):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if "attn2" in name:
                attn_procs[name] = CrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    PnP=PnP,
                    editing_prompts=self.enabled_editing_prompts)
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            #prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = "",
            height: Optional[int] = None,
            width: Optional[int] = None,
            # num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # num_images_per_prompt: int = 1,
            eta: float = 1.0,
            # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            editing_prompt: Optional[Union[str, List[str]]] = None,
            editing_prompt_embeddings: Optional[torch.Tensor] = None,
            reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
            edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
            edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
            edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
            edit_threshold: Optional[Union[float, List[float]]] = 0.9,
            user_mask: Optional[torch.FloatTensor] = None,
            edit_momentum_scale: Optional[float] = 0.1,
            edit_mom_beta: Optional[float] = 0.4,
            edit_weights: Optional[List[float]] = None,
            sem_guidance: Optional[List[torch.Tensor]] = None,
            verbose=True,
            use_cross_attn_mask: bool = False,
            # Attention store (just for visualization purposes)
            attention_store = None,
            text_cross_attention_maps = None,
            attn_store_steps: Optional[List[int]] = [],
            store_averaged_over_steps: bool = True,
            use_intersect_mask: bool = False,
            init_latents = None,
            zs = None,
            max_embeddings_multiples: Optional[int] = 4,
            
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for Semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor>`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as list values should correspond to `editing_prompt`.
                `edit_guidance_scale` is defined as `s_e` of equation 6 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance will not be applied. Momentum
                will still be calculated for those steps and applied once all warmup periods are over.
                `edit_warmup_steps` is defined as `delta` (δ) of [SEGA Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0
                momentum will be disabled. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `sld_warmup_steps`. Momentum will only be added to latent guidance once all warmup periods are
                finished. `edit_momentum_scale` is defined as `s_m` of equation 7 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `edit_warmup_steps`. `edit_mom_beta` is defined as `beta_m` (β) of equation 8 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally. `edit_mom_beta` is defined as `g_i` of equation 9 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Returns:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] if `return_dict` is True,
            otherwise a `tuple. When returning a tuple, the first element is a list with the generated images, and the
            second element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        self.unet.set_attn_processor(AttnProcessor())
        
        # eta = self.eta
        num_images_per_prompt = 1
        # latents = self.init_latents
        latents = init_latents

        use_ddpm = True
        # zs = self.zs
        reset_dpm(self.scheduler)

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = GaussianSmoothing(self.device)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        org_prompt = prompt
        if isinstance(prompt, list):
            assert len(prompt) == self.batch_size
        elif isinstance(prompt, str):
            prompt = list(repeat(prompt, self.batch_size))

        # 2. Define call parameters
        batch_size = self.batch_size

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        # get prompt text embeddings
        # text_inputs = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids
        # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        #         text_input_ids, untruncated_ids
        # ):
        #     removed_text = self.tokenizer.batch_decode(
        #         untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
        #     )
        #     logger.warning(
        #         "The following part of your input was truncated because CLIP can only handle sequences up to"
        #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        #     )

        # text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        # text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings, uncond_embeddings = self._encode_prompt(
                    prompt,
                    self.device,
                    num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=None,
                    max_embeddings_multiples=max_embeddings_multiples,
                )

        if enable_edit_guidance:
            # get safety text embeddings
            if editing_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_length=True
                )

                num_edit_tokens = edit_concepts_input.length - 2  # not counting startoftext and endoftext
                edit_concepts_input_ids = edit_concepts_input.input_ids
                untruncated_ids = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="longest",
                    return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= edit_concepts_input_ids.shape[-1] and not torch.equal(
                        edit_concepts_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                edit_concepts = self.text_encoder(edit_concepts_input_ids.to(self.device))[0]
            else:
                edit_concepts = editing_prompt_embeddings.to(self.device).repeat(batch_size, 1, 1)

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`

        # get unconditional embeddings for classifier free guidance

        if do_classifier_free_guidance:
            # uncond_tokens: List[str]
            # if negative_prompt is None:
            #     uncond_tokens = [""]
            # elif type(prompt) is not type(negative_prompt):
            #     raise TypeError(
            #         f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
            #         f" {type(prompt)}."
            #     )
            # elif isinstance(negative_prompt, str):
            #     uncond_tokens = [negative_prompt]
            # elif batch_size != len(negative_prompt):
            #     raise ValueError(
            #         f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
            #         f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
            #         " the batch size of `prompt`."
            #     )
            # else:
            #     uncond_tokens = negative_prompt

            # max_length = text_input_ids.shape[-1]
            # uncond_input = self.tokenizer(
            #     uncond_tokens,
            #     padding="max_length",
            #     max_length=max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            # seq_len = uncond_embeddings.shape[1]
            # uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            # uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_cross_attention_maps = [org_prompt] if isinstance(org_prompt, str) else org_prompt
            if enable_edit_guidance:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, edit_concepts])
                text_cross_attention_maps += \
                    ([editing_prompt] if isinstance(editing_prompt, str) else editing_prompt)
            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 4. Prepare timesteps
        #self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if use_ddpm:
            t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
            timesteps = timesteps[-zs.shape[0]:]

        if use_cross_attn_mask:
            attention_store = AttentionStore(average=store_averaged_over_steps, batch_size=batch_size)
            self.prepare_unet(attention_store, PnP=False)
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            self.device,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # Initialize edit_momentum to None
        edit_momentum = None

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None
        self.activation_mask = None

        for i, t in enumerate(self.progress_bar(timesteps, verbose=verbose)):
            idx = t_to_idx[int(t)]


            # expand the latents if we are doing classifier free guidance

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * (2 + self.enabled_editing_prompts))
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            text_embed_input = text_embeddings

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample

            # perform guidance
            if do_classifier_free_guidance:

                noise_pred_out = noise_pred.chunk(2 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred_edit_concepts = noise_pred_out[2:]

                # default text guidance
                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.uncond_estimates is None:
                    self.uncond_estimates = torch.zeros((len(timesteps), *noise_pred_uncond.shape))
                self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                if self.text_estimates is None:
                    self.text_estimates = torch.zeros((len(timesteps), *noise_pred_text.shape))
                self.text_estimates[i] = noise_pred_text.detach().cpu()

                if edit_momentum is None:
                    edit_momentum = torch.zeros_like(noise_guidance)

                if sem_guidance is not None and len(sem_guidance) > i:
                    edit_guidance = sem_guidance[i].to(self.device)
                    edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * edit_guidance
                    noise_guidance = noise_guidance + edit_guidance

                elif enable_edit_guidance:
                    if self.activation_mask is None:
                        self.activation_mask = torch.zeros(
                            (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )
                    if self.edit_estimates is None and enable_edit_guidance:
                        self.edit_estimates = torch.zeros(
                            (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )

                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_text.shape))

                    concept_weights = torch.zeros(
                        (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                        device=self.device,
                        dtype=noise_guidance.dtype,
                    )
                    noise_guidance_edit = torch.zeros(
                        (len(noise_pred_edit_concepts), *noise_guidance.shape),
                        device=self.device,
                        dtype=noise_guidance.dtype,
                    )
                    # noise_guidance_edit = torch.zeros_like(noise_guidance)
                    warmup_inds = []
                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        self.edit_estimates[i, c] = noise_pred_edit_concept
                        if isinstance(edit_guidance_scale, list):
                            edit_guidance_scale_c = edit_guidance_scale[c]
                        else:
                            edit_guidance_scale_c = edit_guidance_scale

                        if isinstance(edit_threshold, list):
                            edit_threshold_c = edit_threshold[c]
                        else:
                            edit_threshold_c = edit_threshold
                        if isinstance(reverse_editing_direction, list):
                            reverse_editing_direction_c = reverse_editing_direction[c]
                        else:
                            reverse_editing_direction_c = reverse_editing_direction
                        if edit_weights:
                            edit_weight_c = edit_weights[c]
                        else:
                            edit_weight_c = 1.0
                        if isinstance(edit_warmup_steps, list):
                            edit_warmup_steps_c = edit_warmup_steps[c]
                        else:
                            edit_warmup_steps_c = edit_warmup_steps

                        if isinstance(edit_cooldown_steps, list):
                            edit_cooldown_steps_c = edit_cooldown_steps[c]
                        elif edit_cooldown_steps is None:
                            edit_cooldown_steps_c = i + 1
                        else:
                            edit_cooldown_steps_c = edit_cooldown_steps
                        if i >= edit_warmup_steps_c:
                            warmup_inds.append(c)
                        if i >= edit_cooldown_steps_c:
                            noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                            continue

                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                        # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                        tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                        tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        concept_weights[c, :] = tmp_weights

                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                        if user_mask is not None:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * user_mask

                        if use_cross_attn_mask:
                            out = attention_store.aggregate_attention(
                                attention_maps=attention_store.step_store,
                                prompts=text_cross_attention_maps,
                                res=16,
                                from_where=["up", "down"],
                                is_cross=True,
                                select=text_cross_attention_maps.index(editing_prompt[c]),
                            )
                            attn_map = out[:, :, :, 1:1 + num_edit_tokens[c]]  # 0 -> startoftext

                            # average over all tokens
                            assert (attn_map.shape[3] == num_edit_tokens[c])
                            attn_map = torch.sum(attn_map, dim=3)

                            # gaussian_smoothing
                            attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                            attn_map = self.smoothing(attn_map).squeeze(1)

                            # create binary mask
                            if attn_map.dtype == torch.float32:
                                tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
                            else:
                                tmp = torch.quantile(attn_map.flatten(start_dim=1).to(torch.float32), edit_threshold_c, dim=1).to(attn_map.dtype)
                            attn_mask = torch.where(attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1,16,16), 1.0, 0.0)

                            # resolution must match latent space dimension
                            attn_mask = F.interpolate(
                                attn_mask.unsqueeze(1),
                                noise_guidance_edit_tmp.shape[-2:]  # 64,64
                            ).repeat(1, 4, 1, 1)
                            self.activation_mask[i, c] = attn_mask.detach().cpu()
                            if not use_intersect_mask:
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                        if use_intersect_mask:
                            noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                         keepdim=True)
                            noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp_quantile.dtype)

                            intersect_mask = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                torch.ones_like(noise_guidance_edit_tmp),
                                torch.zeros_like(noise_guidance_edit_tmp),
                            ) * attn_mask

                            self.activation_mask[i, c] = intersect_mask.detach().cpu()

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                        elif not use_cross_attn_mask:
                            # calculate quantile
                            noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                         keepdim=True)
                            noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp_quantile.dtype)

                            self.activation_mask[i, c] = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                torch.ones_like(noise_guidance_edit_tmp),
                                torch.zeros_like(noise_guidance_edit_tmp),
                            ).detach().cpu()

                            noise_guidance_edit_tmp = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                noise_guidance_edit_tmp,
                                torch.zeros_like(noise_guidance_edit_tmp),
                            )

                        noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                    warmup_inds = torch.tensor(warmup_inds).to(self.device)
                    if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                        concept_weights = concept_weights.to("cpu")  # Offload to cpu
                        noise_guidance_edit = noise_guidance_edit.to("cpu")

                        concept_weights_tmp = torch.index_select(concept_weights.to(self.device), 0, warmup_inds)
                        concept_weights_tmp = torch.where(
                            concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                        )
                        concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                        # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                        noise_guidance_edit_tmp = torch.index_select(
                            noise_guidance_edit.to(self.device), 0, warmup_inds
                        )
                        noise_guidance_edit_tmp = torch.einsum(
                            "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                        )
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp
                        noise_guidance = noise_guidance + noise_guidance_edit_tmp

                        self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                        del noise_guidance_edit_tmp
                        del concept_weights_tmp
                        concept_weights = concept_weights.to(self.device)
                        noise_guidance_edit = noise_guidance_edit.to(self.device)

                    concept_weights = torch.where(
                        concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                    )

                    concept_weights = torch.nan_to_num(concept_weights)

                    noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                    noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                    edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                    if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                        noise_guidance = noise_guidance + noise_guidance_edit
                        self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                noise_pred = noise_pred_uncond + noise_guidance

            # compute the previous noisy sample x_t -> x_t-1
            if use_ddpm:
                idx = t_to_idx[int(t)]
                latents = self.scheduler.step(noise_pred, t, latents, variance_noise=zs[idx],
                                              **extra_step_kwargs).prev_sample

            else: #if not use_ddpm:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # step callback
            if use_cross_attn_mask:
                store_step = i in attn_store_steps
                if store_step:
                    print(f"storing attention for step {i}")
                attention_store.between_steps(store_step)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept), attention_store

        return SemanticStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),  attention_store, text_cross_attention_maps

    def encode_text(self, prompts):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        return text_embeddings

    @torch.no_grad()
    def invert(self,
               image_path: str,
               source_prompt: str = "",
               source_guidance_scale=3.5,
               num_inversion_steps: int = 30,
               skip: float = 0.15,
               eta: float = 1.0,
               generator: Optional[torch.Generator] = None,
               verbose=True,
               ):
        """
        Inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        based on the code in https://github.com/inbarhub/DDPM_inversion

        returns:
        zs - noise maps
        xts - intermediate inverted latents
        """

        # self.eta = eta
        # assert (self.eta > 0)
        skip = skip/100
        print("YOOOOOOOOOOOOOOOOO ", skip, num_inversion_steps)
        train_steps = self.scheduler.config.num_train_timesteps
        timesteps = torch.from_numpy(
            np.linspace(train_steps - skip * train_steps - 1, 1, num_inversion_steps).astype(np.int64)).to(self.device)


        self.num_inversion_steps = timesteps.shape[0]
        self.scheduler.num_inference_steps = timesteps.shape[0]
        self.scheduler.timesteps = timesteps
        
        # Reset attn processor, we do not want to store attn maps during inversion
        # self.unet.set_default_attn_processor()
        self.unet.set_attn_processor(AttnProcessor())

        # 1. get embeddings

        uncond_embedding = self.encode_text("")

        # 2. encode image
        x0 = self.encode_image(image_path, dtype=uncond_embedding.dtype)
        self.batch_size = x0.shape[0]

        if not source_prompt == "":
            text_embeddings = self.encode_text(source_prompt).repeat((self.batch_size, 1, 1))
        uncond_embedding = uncond_embedding.repeat((self.batch_size, 1, 1))
        # autoencoder reconstruction
        # image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False)[0]
        # image_rec = self.image_processor.postprocess(image_rec, output_type="pil")
        # 3. find zs and xts
        variance_noise_shape = (
            self.num_inversion_steps,
            self.batch_size,
            self.unet.config.in_channels,
            self.unet.sample_size,
            self.unet.sample_size)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = self.num_inversion_steps-t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, t)
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        reset_dpm(self.scheduler)
        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        for t in self.progress_bar(timesteps, verbose=verbose):

            idx = self.num_inversion_steps-t_to_idx[int(t)]-1
            # 1. predict noise residual
            xt = xts[idx+1]

            noise_pred = self.unet(xt, timestep=t, encoder_hidden_states=uncond_embedding).sample

            if not source_prompt == "":
                noise_pred_cond = self.unet(xt, timestep=t, encoder_hidden_states=text_embeddings).sample
                noise_pred = noise_pred + source_guidance_scale * (noise_pred_cond - noise_pred)

            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        # TODO: I don't think that the noise map for the last step should be discarded ?!
        # if not zs is None:
        #     zs[-1] = torch.zeros_like(zs[-1])
        # self.init_latents = xts[-1].expand(self.batch_size, -1, -1, -1)
        zs = zs.flip(0)
        # self.zs = zs


        return zs, xts
        # return zs, xts, image_rec

    @torch.no_grad()
    def encode_image(self, image_path, dtype=None):
        image = load_512(image_path,
                         size=self.unet.sample_size * self.vae_scale_factor,
                         device=self.device,
                         dtype=dtype)
        x0 = self.vae.encode(image).latent_dist.mode()
        x0 = self.vae.config.scaling_factor * x0
        return x0

def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * noise_pred

    # modifed so that updated xtm1 is returned as well (to avoid error accumulation)
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)

    return noise, mu_xt + (eta * variance ** 0.5) * noise

# Copied from pipelines.StableDiffusion.CycleDiffusionPipeline.compute_noise
def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):

    def first_order_update(model_output, timestep, prev_timestep, sample):
        lambda_t, lambda_s = scheduler.lambda_t[prev_timestep], scheduler.lambda_t[timestep]
        alpha_t, alpha_s = scheduler.alpha_t[prev_timestep], scheduler.alpha_t[timestep]
        sigma_t, sigma_s = scheduler.sigma_t[prev_timestep], scheduler.sigma_t[timestep]
        h = lambda_t - lambda_s

        mu_xt = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample
    def second_order_update(model_output_list, timestep_list, prev_timestep, sample):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = scheduler.lambda_t[t], scheduler.lambda_t[s0], scheduler.lambda_t[s1]
        alpha_t, alpha_s0 = scheduler.alpha_t[t], scheduler.alpha_t[s0]
        sigma_t, sigma_s0 = scheduler.sigma_t[t], scheduler.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        mu_xt = (
            (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    step_index = (scheduler.timesteps == timestep).nonzero()
    if len(step_index) == 0:
        step_index = len(scheduler.timesteps) - 1
    else:
        step_index = step_index.item()

    prev_timestep = 0 if step_index == len(scheduler.timesteps) - 1 else scheduler.timesteps[step_index + 1]

    model_output = scheduler.convert_model_output(noise_pred, timestep, latents)

    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, timestep, prev_timestep, latents)
    else:
        timestep_list = [scheduler.timesteps[step_index - 1], timestep]
        noise, prev_sample = second_order_update(scheduler.model_outputs, timestep_list, prev_timestep, latents)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    return noise, prev_sample

def compute_noise(scheduler, *args):
    if isinstance(scheduler, DDIMScheduler):
        return compute_noise_ddim(scheduler, *args)
    elif isinstance(scheduler, DPMSolverMultistepSchedulerInject) and scheduler.config.algorithm_type == 'sde-dpmsolver++'\
            and scheduler.config.solver_order == 2:
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        raise NotImplementedError