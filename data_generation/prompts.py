"""Prompt library for VDE-Bench data generation.

These prompts are passed to a vision-capable LLM (e.g. gemini-3-pro-image) to
either (a) generate an editing instruction for a document image, or (b) apply
an editing instruction and return a modified image.

All prompts are plain-English strings with no model-specific formatting so
they can be reused across different OpenAI-compatible backends.
"""

# ---------------------------------------------------------------------------
# Legacy short-form prompts (used by the first-iteration crop-based pipeline).
# They return the modified image + a free-form instruction in a JSON blob.
# ---------------------------------------------------------------------------

PROMPT_TABLE_DELETE = """Randomly delete the text in the input table. Then return the modification instruction and the modified image. Note that under all circumstances the output image must have the same resolution as the original image.
    The return format follows the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """

PROMPT_TABLE_ADD = """Randomly add some text in the input table. Then return the modification instruction and the modified image. Note that under all circumstances the output image must have the same resolution as the original image.
    The return format follows the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """

PROMPT_TABLE_COLOR = """Modify the colors of the table. Then return the modification instructions as well as the modified image. Please note that under all circumstances, the resolution of the output image must remain the same as the original image. The return format should follow the following JSON structure: { "instruction": "modification instruction" }"""


PROMPT_FIGURE_TEXT = """Randomly modify the text in the input figure. Then return the modification instruction and the modified image. Note that under all circumstances the output image must have the same resolution as the original image.
    The return format follows the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """

PROMPT_FIGURE_COLOR = """Modify the color of the figure. Then return the modification instruction and the modified image. Note that under all circumstances the output image must have the same resolution as the original image.
    The return format follows the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """

PROMPT_FIGURE_TYPE = """Modify the type of the input figure. Then return the modification instruction and the modified image. Note that under all circumstances the output image must have the same resolution as the original image.
    The return format follows the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """

PROMPT_TEXT_GENERIC = """Randomly modify the text in the input image. Then return the modification instruction and the modified image. Note that under all circumstances, the output image must have the same resolution as the original image. In addition, do not modify any text inside tables; only titles, body text, and other non-table content may be changed.
    The return format should follow the JSON structure below:
    {
        "instruction": "Modification instruction"
    }
    """


# ---------------------------------------------------------------------------
# Instruction-generation prompts (first-pass). Produce a *text* instruction
# given an input image; the instruction is then applied in a second stage.
# ---------------------------------------------------------------------------

INSTR_TEXT_DELETE = """Generate a **text deletion instruction** for the input image. You are not allowed to modify any text inside tables or within images; only titles and body text may be modified.
        1. Your response must contain only the editing instruction itself, with no additional content.
        2. Your response must be plain text, without any Markdown formatting.
        3. The instruction you provide must clearly specify which text in the image is to be deleted.
        4. The language of your instruction must match the primary language used in the image. For example, if the main language in the image is Chinese, respond in Chinese; if it is English, respond in English.
        5. Modify only one location.
"""

INSTR_TEXT_ADD = """Generate a **text addition instruction** for the input image. You are not allowed to modify any text inside tables or within images; only titles and body text may be modified.
    1. Your response must contain only the editing instruction itself, with no additional content.
    2. Your response must be plain text, without any Markdown formatting.
    3. The instruction you provide must clearly specify which text in the image is to be added.
    4. The language of your instruction must match the primary language used in the image. For example, if the main language in the image is Chinese, respond in Chinese; if it is English, respond in English.
    5. Modify only one location.
"""

INSTR_TABLE_DELETE = """Generate a **text deletion instruction** for the input image. You are not allowed to modify any text inside tables or within images; only titles and body text may be modified.
        1. Your response must contain only the editing instruction itself, with no additional content.
        2. Your response must be plain text, without any Markdown formatting.
        3. The instruction you provide must clearly specify which text in the image is to be deleted.
        4. The language of your instruction must match the primary language used in the image. For example, if the main language in the image is Chinese, respond in Chinese; if it is English, respond in English.
        5. Modify only one location.
**You may only -- and must -- modify the table portion of the image.**
"""

INSTR_TABLE_ADD = """Generate a **text addition instruction** for the input image. You are not allowed to modify any text inside tables or within images; only titles and body text may be modified.
    1. Your response must contain only the editing instruction itself, with no additional content.
    2. Your response must be plain text, without any Markdown formatting.
    3. The instruction you provide must clearly specify which text in the image is to be added.
    4. The language of your instruction must match the primary language used in the image. For example, if the main language in the image is Chinese, respond in Chinese; if it is English, respond in English.
    5. Modify only one location.
**You may only -- and must -- modify the table portion of the image.**
"""

INSTR_TABLE_STRUCTURE = """Generate a **table structure modification instruction** for the input image, such as merging rows or merging columns, but do not adjust spacing or layout. You are not allowed to modify any text inside tables or any text within images; only titles and body text may be modified.
1. Your response must contain only the editing instruction itself, with no additional content.
2. Your response must be plain text, without any Markdown formatting.
3. The instruction you provide must clearly specify where in the image the modification should be made.
4. The language of your instruction must match the primary language used in the image. For example, if the main language in the image is Chinese, respond in Chinese; if it is English, respond in English.
5. Modify only one location.
**You may only -- and must -- modify the table portion of the image.**
"""


# ---------------------------------------------------------------------------
# Inference prompts -- used to *reverse-engineer* an instruction from a
# (before, after) image pair, optionally guided by a local reference.
# ---------------------------------------------------------------------------

INFER_INSTRUCTION_PROMPT = """Here are two images: the first is the original image, and the second is the image after being modified by an image-editing instruction. Please infer the editing instruction based on the differences between the two images.
        1. Your response should contain nothing other than the instruction itself.
        2. Your response should be plain text only, without any markdown formatting.
        3. The instruction you provide must clearly specify which table, image, or text in the picture was modified.
        4. The language of your instruction must match the primary language used in the images. For example, if the main language in the images is Chinese, respond in Chinese; if the main language is English, respond in English.
        5. There may be a local editing instruction that is correct but only focuses on a specific part of the two images. If such a local editing instruction is provided, you should refer to it.

        Local editing instruction: {reference}

"""


CLASSIFY_INSTRUCTION_PROMPT = """Based on the instruction content, return the type of the instruction.  The type must be one of [text deletion, text addition, table structure edit].
    1. Your response can only be one of these three values with no additional output.
    Instruction: {instruction}
"""


__all__ = [
    "PROMPT_TABLE_DELETE",
    "PROMPT_TABLE_ADD",
    "PROMPT_TABLE_COLOR",
    "PROMPT_FIGURE_TEXT",
    "PROMPT_FIGURE_COLOR",
    "PROMPT_FIGURE_TYPE",
    "PROMPT_TEXT_GENERIC",
    "INSTR_TEXT_DELETE",
    "INSTR_TEXT_ADD",
    "INSTR_TABLE_DELETE",
    "INSTR_TABLE_ADD",
    "INSTR_TABLE_STRUCTURE",
    "INFER_INSTRUCTION_PROMPT",
    "CLASSIFY_INSTRUCTION_PROMPT",
]
