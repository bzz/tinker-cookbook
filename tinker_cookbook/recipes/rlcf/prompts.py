"""
Prompts for the RLCF (Reinforcement Learning from Checklist Feedback) pipeline.

All prompts are faithfully reproduced from the reference implementation at
https://github.com/viswavi/RLCF (Viswanathan et al., 2025).
"""

import re

# ---------------------------------------------------------------------------
# Checklist generation prompt (no-candidates variant for online use)
# ---------------------------------------------------------------------------

CHECKLIST_GENERATION_PROMPT = """You are responsible for developing criteria for judging arbitrary responses to instructions. You will be given an instruction (the kind given to AI assistants like yourself), and your goal is to write a list of criteria-style questions that must be satisfied by any valid response to the instruction. In addition to the instruction, you will also be a response written by an (imperfect) expert for comparison. You will generate the criteria questions by identifying clear, measurable ways in which potential responses may deviate from the given instructions. First, describe your reasoning, then produce a response containing a list of questions. For each question, weight the importance of the question from 0 to 100. 100 indicates a question that is absolutely critical to the validity of the response. 75 indicates a question that is critical to response quality but may not be explicitly stated by the instruction. 50 indicates a question that should be answered by any good response, but a response could still be useful without this question being answered. 25 indicates a question that is a preference but not a requirement. Less than 25 indicates a question that is not important to the validity of the response (e.g. a soft nice-to-have).

Your Task:
1. Carefully examine the original instruction
2. Describe your reasoning in identifying specific, objective criteria from the instruction that any response should satisfy
3. Write concise questions that must be satisfied by any valid response.
4. Weight the importance of each question from 0 to 100.

Question Guidelines:
- Each question should test exactly ONE requirement
- Questions should be easily verifiable, almost as if writing a Boolean condition in Python
- Frame questions to require clear yes/no answers
- Focus only on objective, measurable criteria
- Return "None" if there are no obvious requirements to extract
- Weight each question from 0 to 100 based on its importance.

Formatting:
- Format as a bulleted list. Each question should start with "- "
- Conclude by writing "<END>" after finishing your bulleted list
- Do not provide any explanations after writing your questions; just write the bulleted list of questions and weights then write "<END>"
- Start by writing "Key Criteria Questions:"
- Most questions should start with "Does", "Do", "Is", or "Are"
- Use concise and unambiguous language
- Phrase requirements positively (avoid "not" or negatives)
- Include only a few questions (between 1 and 7, in most cases).
- For each question, include the weight of this question in parentheses after the question mark e.g. "(100)"

Let's take an example instruction: "Write a tweet about cats using exactly 280 characters"

Here are some bad questions:
- Is the generated text interesting? - This is subjective
- Does the generated text discuss cats in fewer than 280 characters? - This question overloads multiple aspects
- Is the generated text not about dogs? - This question uses negative phrasing
- Is the generated text helpful and harmless - This question is overly general

Key Criteria Questions:
- Is the generated text about cats? (100)
- Does the generated text contain exactly 280 characters? (95)
- Is the generated text written in a casual, social media-friendly tone? (70)
<END>

Instruction:
"System: Summarize the movie in a snarky way. Try to explain the movie in just one sentence.
User: The Shining"

Expert Response:
"A family moves into a haunted hotel for the winter, where dad goes crazy from writer's block, ghosts, and no Twitter - but at least the kid gets to ride his bike through creepy hallways."

Reasoning:
The instruction explicitly asks for a summary. The instruction also asks for the summary to be snarky, and the instruction asks for the summary to try to be one sentence long. The expert response satisfies all of these criteria. The text being a summary of the movie (The Shining) is an absolute necessity, which we will weigh as 100/100 points. The response being snarky is also a very important, but slightly less so, so we can weigh it as 95/100 points. The response being only one sentence is also crucial but the response could still be useful if this is loosely violated, so we can weigh it as 80/100 points.

Key Criteria Questions:
- Is the generated text the summary of a movie (The Shining)? (100)
- Is the generated summary written in a snarky way? (95)
- Does the generated summary only contain one sentence? (80)
<END>

Instruction:
"System: Extract the address of the property from the "About this space" section of Airbnb.
User: Tucked in the foothills of the quaint historic mining town of Grass Valley, CA this funky and spacious chalet invites you to experience the great outdoors with family, friends and pets. Enjoy the hot tub, basketball court, treehouse, kids rooms, bbq, fire-pit, outdoor theater, and more. One hour from Tahoe and 10 minutes to town, you're never short of things to explore."

Expert Response:
"The address is not specified in the given text."

Reasoning:
The instruction explicitly asks for an address, extracted from a description of a property. In addition to providing an address, a correct response must explicitly specified in the text contained in the given input - any other address would be incorrect. Both of these are absolutely critical requirements and will be given full weight (100/100).

Key Criteria Questions:
- Is the generated text an address? (100)
- Is the generated text the address of the property according to the text in the given input? (100)"
<END>

Instruction:
"{instruction}"

Expert Response:
"{expert_response}"

Reasoning:"""

# ---------------------------------------------------------------------------
# Checklist evaluation prompt — numerical scoring (0–100)
# Faithfully reproduced from RLCF construct_offline_preference_data.py
# ---------------------------------------------------------------------------

CHECKLIST_EVAL_NUMERICAL_PROMPT = """"Based on the provided input instruction and response from a worker, assess the response based on the following single-criterion question. Score the response with a rating (a number between 0 and 100) assessing how well the response answers that question. For example, the input instruction might be "What is a good vegan substitute to meat for someone allergic to soy and gluten? Provide an answer followed by a factually detailed and humorous one-sentence explanation" and the criterion question might be "Is the explanation factually detailed?". Your selection should be based primarily on the response and the question alone, with the instruction shown for context when needed:
- 100: Select 100 if the generated text represents an optimal solution that expertly balances all relevant factors mentioned in the question. For objective criteria (like "Does each sentence in the generated text use a second person?"), even minor deviations exclude a 100 rating (and probably lead to a 0 rating). For subjective criteria, the response should basically be perfect. For the example above (about the vegan substitute), and the criterion above (about factual detail), an example 100-point response is "Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they can be cooked into crispy OR chewy meatlike textures.". This response is richly detailed and factual, and though it fails to be humorous, it is still a 100-point response on the factual detail criterion.
- 75: Return ~75 if the generated text very effectively addresses the main requirements but has room for minor improvements. The response should be unconditionally acceptable (at a professional level) but may not be absolutely perfect. There are no mistakes that critically undermine the question. An example 75-point response to the example question above is "Mushrooms - they are rich in the glutamates that lead to incredible umami flavors and they don't look cute in the slightest while alive.". This response has one interesting fact but could be more detailed.
- 50: Opt for 50 if the generated text adequately fulfills the basic requirements but contains notable flaws or missed opportunities for improvement. The response should still be functionally acceptable. The response contains at most one minor inadequacy or inaccuracy related to the question but there are no mistakes that critically undermine the question. An example 50-point response to the example question above is "Mushrooms, because they can be easily caramelized and browned, they're universally beloved by sophisticated palates, and they don't look cute in the slightest while alive." The statement that they're universally beloved by people with sophisticated palates, while potentially true, is vague and not objective.
- 25: Return ~25 if the generated text fulfills the key condition specified by the question and demonstrates awareness of the key requirements but fails to execute them effectively. The text may contain non-critical inaccuracies or irrelevant information. However, if there is even one element that critically undermines the core purpose specified in the question (even if that element seems minor in isolation), the score should be 0 (not 25). An example 25-point response to the example question above is "Mushrooms, because they can be easily caramelized and browned, universally beloved by kids, and they don't look cute in the slightest while alive." The statement that most kids love mushrooms is not objective and potentially false).
- 0: Opt for 0 if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. If the response contains a critical error relevant to the question, return a 0. For the question about the vegan substitute, an example 0-point response is "Mushrooms, because they make you question why you ever thought a dead animal could compare to this vegan delight." While funny and engaging, this response contains zero factual detail about mushrooms, critically violating the question.

Your score can be any number between 0 and 100 (not just the ones listed above). If you are totally confused, return -1 as a default. You should use your judgment to determine the most appropriate score. Focus on the posed question and ignore other aspects of response quality not implied by the question. Return only a number - do not include any other text in your response.

Input:
{instruction}

Generated Text:
{response}

Question:
{requirement}

Score: """

# ---------------------------------------------------------------------------
# Universal quality requirements (appended to every checklist)
# ---------------------------------------------------------------------------

UNIVERSAL_REQUIREMENT_TEXT = (
    "Does the response satisfy the following two criteria:\n"
    "1) The response directly address the request without excessive or "
    "off-topic information not necessary for addressing the user's instruction?\n"
    "2) The response should match the context and the instruction, whether it "
    "requires professionalism, friendliness, formality, or neutrality."
)
UNIVERSAL_REQUIREMENT_WEIGHT = 100


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_checklist_response(response_text: str) -> list[tuple[str, float]]:
    """Parse a checklist-generation LLM response into (requirement, weight) pairs.

    Handles the structured format output by the checklist generation prompt:
    ``Key Criteria Questions:``
    ``- Does the text ...? (100)``
    ``<END>``
    """
    text = response_text
    if "<END>" in text:
        text = text.split("<END>")[0].strip()
    if "Key Criteria Questions:" in text:
        text = text.split("Key Criteria Questions:")[-1].strip()

    requirements: list[tuple[str, float]] = []
    if text.strip().lower() == "none":
        return requirements

    lines = re.split(r"(?:^|\n)\s*[-*]\s+", text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.strip().lower() == "none":
            return []
        match = re.search(r"\((\d+(?:\.\d+)?)\)\s*$", line)
        if match:
            weight = float(match.group(1))
            requirement = line[: match.start()].strip()
            if requirement and not requirement.startswith("Here are"):
                requirements.append((requirement, weight))
    return requirements


def format_eval_prompt(instruction: str, response: str, requirement: str) -> str:
    """Format the numerical (0-100) checklist evaluation prompt."""
    return CHECKLIST_EVAL_NUMERICAL_PROMPT.format(
        instruction=instruction,
        response=response,
        requirement=requirement,
    )


def parse_eval_score(text: str) -> float:
    """Extract a numerical score from the judge LLM's response.

    Returns a score in [0, 100], or -1 if parsing fails.
    """
    text = text.strip()
    first_token = text.split()[0] if text.split() else ""
    first_token = first_token.rstrip(".,;:)")
    try:
        score = float(first_token)
        return max(0.0, min(100.0, score))
    except ValueError:
        return -1.0
