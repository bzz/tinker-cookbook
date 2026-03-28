"""
Generate example RLCF training / test data.

Creates synthetic datapoints for testing the RLCF pipeline.
For real training, use ``ChecklistDatapointListBuilderFromHF`` to load
pre-computed checklists from ``viswavi/rlcf``.
"""

import random
from pathlib import Path

from tinker_cookbook.recipes.rlcf.data import ChecklistDatapoint, ChecklistItem


_EXAMPLE_INSTRUCTIONS = [
    (
        "Write a haiku about the ocean.",
        [
            ChecklistItem("Does the text follow a 5-7-5 syllable pattern?", 100),
            ChecklistItem("Is the text about the ocean?", 100),
            ChecklistItem("Is the text exactly 3 lines long?", 90),
        ],
    ),
    (
        "Explain what a black hole is in exactly two sentences, suitable for a 10-year-old.",
        [
            ChecklistItem("Is the explanation about black holes?", 100),
            ChecklistItem("Does the response contain exactly two sentences?", 95),
            ChecklistItem("Is the language simple enough for a 10-year-old?", 80),
        ],
    ),
    (
        "List 3 benefits of regular exercise. Use bullet points.",
        [
            ChecklistItem("Does the response list exactly 3 benefits?", 100),
            ChecklistItem("Are the benefits about regular exercise?", 100),
            ChecklistItem("Does the response use bullet points?", 90),
        ],
    ),
    (
        "Write a professional email to a colleague declining a meeting invitation for Friday.",
        [
            ChecklistItem("Is the text formatted as an email?", 100),
            ChecklistItem("Does the email decline a meeting invitation?", 100),
            ChecklistItem("Does the email mention Friday?", 85),
            ChecklistItem("Is the tone professional?", 75),
        ],
    ),
    (
        "Translate 'Good morning, how are you?' to French.",
        [
            ChecklistItem("Does the response contain a French translation?", 100),
            ChecklistItem("Is the translation of the phrase 'Good morning, how are you?'?", 100),
        ],
    ),
    (
        "Write a short recipe for scrambled eggs. Include ingredients and steps.",
        [
            ChecklistItem("Is the recipe for scrambled eggs?", 100),
            ChecklistItem("Does the recipe include a list of ingredients?", 95),
            ChecklistItem("Does the recipe include step-by-step instructions?", 95),
        ],
    ),
    (
        "Summarize the plot of Romeo and Juliet in 3 sentences or fewer.",
        [
            ChecklistItem("Is the summary about Romeo and Juliet?", 100),
            ChecklistItem("Does the summary contain 3 sentences or fewer?", 90),
            ChecklistItem("Does the summary cover the main plot points?", 75),
        ],
    ),
    (
        "Write a limerick about a cat.",
        [
            ChecklistItem("Does the text follow the AABBA rhyme scheme of a limerick?", 100),
            ChecklistItem("Is the text about a cat?", 100),
            ChecklistItem("Is the text exactly 5 lines long?", 90),
        ],
    ),
    (
        "Explain the difference between a virus and a bacterium in simple terms.",
        [
            ChecklistItem("Does the response explain what a virus is?", 100),
            ChecklistItem("Does the response explain what a bacterium is?", 100),
            ChecklistItem("Does the response highlight the differences between the two?", 90),
            ChecklistItem("Is the language simple and accessible?", 70),
        ],
    ),
    (
        "Create a to-do list for planning a birthday party. Include at least 5 items.",
        [
            ChecklistItem("Is the response a to-do list?", 100),
            ChecklistItem("Is the list about planning a birthday party?", 100),
            ChecklistItem("Does the list contain at least 5 items?", 95),
        ],
    ),
]


def generate_one(rng: random.Random) -> ChecklistDatapoint:
    instruction, checklist = rng.choice(_EXAMPLE_INSTRUCTIONS)
    return ChecklistDatapoint(instruction=instruction, checklist=checklist)


def generate_dataset(
    num_train: int,
    num_test: int,
    seed: int,
    write_dir: str = "tinker_cookbook/example_data/",
) -> tuple[str, str]:
    rng = random.Random(seed)
    all_datapoints = [generate_one(rng) for _ in range(num_train + num_test)]

    write_path = Path(write_dir)
    write_path.mkdir(parents=True, exist_ok=True)

    train_path = str(write_path / "example_rlcf_train.jsonl")
    with open(train_path, "w") as f:
        for dp in all_datapoints[:num_train]:
            f.write(dp.to_json() + "\n")
    print(f"Generated {num_train} train datapoints in {train_path}")

    test_path = str(write_path / "example_rlcf_test.jsonl")
    with open(test_path, "w") as f:
        for dp in all_datapoints[num_train:]:
            f.write(dp.to_json() + "\n")
    print(f"Generated {num_test} test datapoints in {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    generate_dataset(num_train=10000, num_test=1000, seed=42)
