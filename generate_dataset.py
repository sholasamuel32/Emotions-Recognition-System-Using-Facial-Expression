"""Generate the balanced demonstration dataset used by MoodSense AI.

The data are synthetic and curated for an educational ML project. They must not
be represented as clinical observations or used to diagnose a person.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


MOOD_PHRASES = {
    "happy": [
        "I feel genuinely happy and thankful",
        "my heart feels light and full of joy",
        "I am pleased with how things turned out",
        "I keep smiling because today has been good",
        "I feel proud, cheerful, and satisfied",
        "this has filled me with gratitude",
        "I am delighted by the result",
        "I feel loved and appreciated",
        "everything seems bright and hopeful",
        "I enjoyed the moment more than I expected",
        "I feel blessed by the support around me",
        "the good news has lifted my spirit",
        "I am content and in a positive mood",
        "I feel glad that my effort paid off",
        "today brought me a quiet kind of happiness",
        "I am celebrating a meaningful achievement",
    ],
    "calm": [
        "I feel peaceful and completely at ease",
        "my thoughts are clear and settled",
        "I am relaxed and breathing steadily",
        "everything feels quiet and manageable",
        "I feel balanced and emotionally steady",
        "there is no rush and I am comfortable",
        "I have accepted what I cannot control",
        "my mind feels still after a long day",
        "I am composed and taking things slowly",
        "the silence around me feels soothing",
        "I feel safe, grounded, and present",
        "I am resting without any pressure",
        "my body and mind both feel relaxed",
        "I feel patient with the process",
        "I am comfortable with my decision",
        "the gentle routine has helped me settle",
    ],
    "excited": [
        "I cannot wait for what is coming next",
        "I am thrilled about this new opportunity",
        "I have so much energy and enthusiasm",
        "this idea has me buzzing with anticipation",
        "I am eager to begin the new project",
        "the big day is finally almost here",
        "I feel inspired and ready to act",
        "this surprise is absolutely amazing",
        "I am looking forward to the celebration",
        "I can hardly contain my enthusiasm",
        "I feel pumped about the challenge",
        "my plans for tomorrow sound incredible",
        "I am counting down to the event",
        "this possibility feels fresh and energising",
        "I am enthusiastic about meeting the team",
        "I feel a strong burst of positive energy",
    ],
    "anxious": [
        "I am worried that something may go wrong",
        "my thoughts keep racing and I cannot relax",
        "I feel nervous about the outcome",
        "the uncertainty is making me uneasy",
        "I am overwhelmed by all these deadlines",
        "I keep imagining the worst possible result",
        "the pressure is making it hard to focus",
        "I feel tense and restless tonight",
        "I am scared that I will make a mistake",
        "my stomach is in knots from worry",
        "I do not know how I will cope with this",
        "I feel panicked whenever I think about it",
        "I have been overthinking every small detail",
        "I cannot stop worrying about tomorrow",
        "my mind feels crowded with fearful thoughts",
        "I feel uncertain and on edge",
    ],
    "sad": [
        "I feel lonely and emotionally empty",
        "my heart feels heavy after what happened",
        "I am disappointed and deeply hurt",
        "I miss the way things used to be",
        "I feel low and have very little energy",
        "the loss has left me heartbroken",
        "I want to cry when I remember it",
        "I feel forgotten and alone",
        "nothing seems to lift my mood today",
        "I have been feeling down all week",
        "the bad news has taken away my motivation",
        "I feel sorrowful and withdrawn",
        "today has been painful and difficult",
        "I am struggling with a sense of loss",
        "I feel discouraged by the setback",
        "the disappointment still makes me tearful",
    ],
    "angry": [
        "I am furious about how I was treated",
        "this decision is completely unfair",
        "I am annoyed that the problem happened again",
        "their disrespect has made me very angry",
        "I cannot stand being ignored like this",
        "the situation is making me lose my temper",
        "I am irritated by their careless behaviour",
        "I hate being blamed for someone else's mistake",
        "I feel frustrated and ready to shout",
        "their rude response made me furious",
        "I have had enough of these excuses",
        "this is unacceptable and deeply upsetting",
        "I resent the way they dismissed my concerns",
        "I am mad that nobody listened",
        "the broken promise has made me bitter",
        "I feel provoked by the repeated disrespect",
    ],
    "neutral": [
        "the meeting starts at four tomorrow",
        "I submitted the form at the main office",
        "the package arrived on Tuesday afternoon",
        "there are six items on the current list",
        "I travelled to work by bus this morning",
        "the report contains five main sections",
        "I had rice and vegetables for lunch",
        "the class will continue again next week",
        "I completed the usual daily tasks",
        "this message is an update about the project",
        "the appointment is scheduled for Monday",
        "I am currently sitting in the living room",
        "the document was saved in the shared folder",
        "the shop closes at seven in the evening",
        "I will call the office after breakfast",
        "the blue notebook is beside the computer",
    ],
}

OPENERS = [
    "Right now,", "Honestly,", "At the moment,", "Today,", "This evening,",
    "When I think about everything,", "After the recent events,", "For me,",
    "Since this morning,", "At this point,", "In this situation,", "Lately,",
]

CONTEXTS = [
    "because of what happened at work",
    "after speaking with my family",
    "while thinking about tomorrow",
    "as I reflect on the whole situation",
    "following the message I received",
    "because the plan changed unexpectedly",
    "after completing my responsibilities",
    "when I consider the next step",
    "because of the conversation we had",
    "after spending some time alone",
    "as the day comes to an end",
    "while preparing for the coming week",
]

ENDINGS = [
    "That is the best way I can describe it.",
    "The feeling has stayed with me for hours.",
    "I wanted to put the experience into words.",
    "It is affecting the way I see the day.",
    "I have noticed the feeling quite clearly.",
    "This is what has been on my mind.",
    "I am trying to understand the feeling better.",
    "It has shaped my mood today.",
]


def generate_rows(per_class: int = 280, seed: int = 42) -> list[dict[str, str]]:
    """Return deterministic, balanced, unique text/label rows."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for mood, phrases in MOOD_PHRASES.items():
        candidates: set[str] = set()
        while len(candidates) < per_class:
            opener = rng.choice(OPENERS)
            phrase = rng.choice(phrases)
            context = rng.choice(CONTEXTS)
            ending = rng.choice(ENDINGS)
            text = f"{opener} {phrase} {context}. {ending}"
            roll = rng.random()
            if roll < 0.06:
                text = text.lower()
            elif roll < 0.12:
                text = text.replace(",", "").replace(".", "")
            candidates.add(text)
        rows.extend(
            {"text": text, "mood": mood, "source": "synthetic_curated_v1"}
            for text in sorted(candidates)
        )
    rng.shuffle(rows)
    return rows


def write_dataset(output: Path, per_class: int = 280, seed: int = 42) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(per_class=per_class, seed=seed)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "mood", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the MoodSense dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/mood_text_dataset.csv"))
    parser.add_argument("--per-class", type=int, default=280)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    destination = write_dataset(args.output, args.per_class, args.seed)
    print(f"Dataset written to {destination} ({args.per_class * len(MOOD_PHRASES):,} rows)")
