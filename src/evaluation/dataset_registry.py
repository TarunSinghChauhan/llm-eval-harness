import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


class DatasetRegistry:
    """
    Versioned dataset registry.
    Datasets are stored as JSON, hashed for reproducibility.
    """

    def load(self, name: str, version: str = "v1") -> dict[str, Any]:
        path = DATASETS_DIR / f"{name}_{version}.json"
        if not path.exists():
            # Auto-generate built-in datasets
            if name == "mmlu_sample":
                return self._create_mmlu_sample(version)
            elif name == "reasoning":
                return self._create_reasoning_dataset(version)
            elif name == "instruction_following":
                return self._create_instruction_dataset(version)
            raise FileNotFoundError(f"Dataset not found: {name} {version}")

        with open(path) as f:
            data = json.load(f)

        logger.info("dataset_loaded", name=name, version=version, n_prompts=len(data["prompts"]))
        return data

    def _save(self, name: str, version: str, data: dict) -> dict:
        data["version"] = version
        data["hash"] = hashlib.sha256(
            json.dumps(data["prompts"], sort_keys=True).encode()
        ).hexdigest()[:12]
        path = DATASETS_DIR / f"{name}_{version}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("dataset_saved", name=name, version=version, hash=data["hash"])
        return data

    def _create_mmlu_sample(self, version: str) -> dict:
        """50 MMLU-style multiple choice questions across domains."""
        prompts = [
            # Science
            {"id": "sci_001", "prompt": "What is the speed of light in a vacuum?", "reference": "approximately 299,792,458 meters per second"},
            {"id": "sci_002", "prompt": "What particle has a negative charge and orbits the nucleus?", "reference": "electron"},
            {"id": "sci_003", "prompt": "What is the chemical formula for water?", "reference": "H2O"},
            {"id": "sci_004", "prompt": "What force keeps planets in orbit around the sun?", "reference": "gravity"},
            {"id": "sci_005", "prompt": "What is the powerhouse of the cell?", "reference": "mitochondria"},
            {"id": "sci_006", "prompt": "What is the atomic number of carbon?", "reference": "6"},
            {"id": "sci_007", "prompt": "What gas do plants absorb during photosynthesis?", "reference": "carbon dioxide"},
            {"id": "sci_008", "prompt": "What is Newton's second law of motion?", "reference": "force equals mass times acceleration"},
            {"id": "sci_009", "prompt": "What is the half-life concept in radioactive decay?", "reference": "the time required for half of the radioactive atoms in a sample to decay"},
            {"id": "sci_010", "prompt": "What is absolute zero in Celsius?", "reference": "-273.15 degrees Celsius"},
            # Math
            {"id": "math_001", "prompt": "What is the derivative of x squared?", "reference": "2x"},
            {"id": "math_002", "prompt": "What is the Pythagorean theorem?", "reference": "a squared plus b squared equals c squared"},
            {"id": "math_003", "prompt": "What is the value of pi to 5 decimal places?", "reference": "3.14159"},
            {"id": "math_004", "prompt": "What is the integral of 1/x?", "reference": "natural log of x plus a constant"},
            {"id": "math_005", "prompt": "What is Euler's number e to 4 decimal places?", "reference": "2.7183"},
            {"id": "math_006", "prompt": "How many sides does a dodecagon have?", "reference": "12"},
            {"id": "math_007", "prompt": "What is the sum of angles in a triangle?", "reference": "180 degrees"},
            {"id": "math_008", "prompt": "What is the formula for the area of a circle?", "reference": "pi times radius squared"},
            {"id": "math_009", "prompt": "What is the square root of 144?", "reference": "12"},
            {"id": "math_010", "prompt": "What does the quadratic formula solve for?", "reference": "the roots or solutions of a quadratic equation"},
            # History
            {"id": "hist_001", "prompt": "In what year did World War 2 end?", "reference": "1945"},
            {"id": "hist_002", "prompt": "Who was the first President of the United States?", "reference": "George Washington"},
            {"id": "hist_003", "prompt": "What empire did Julius Caesar represent?", "reference": "Roman Empire"},
            {"id": "hist_004", "prompt": "In what year did the Berlin Wall fall?", "reference": "1989"},
            {"id": "hist_005", "prompt": "What was the name of the first artificial satellite launched into space?", "reference": "Sputnik"},
            {"id": "hist_006", "prompt": "Who wrote the Declaration of Independence?", "reference": "Thomas Jefferson was the primary author"},
            {"id": "hist_007", "prompt": "What year did the French Revolution begin?", "reference": "1789"},
            {"id": "hist_008", "prompt": "Which country first gave women the right to vote nationally?", "reference": "New Zealand in 1893"},
            {"id": "hist_009", "prompt": "What was the name of the ship that sank in 1912?", "reference": "Titanic"},
            {"id": "hist_010", "prompt": "In what year did India gain independence from Britain?", "reference": "1947"},
            # Computer Science
            {"id": "cs_001", "prompt": "What does CPU stand for?", "reference": "Central Processing Unit"},
            {"id": "cs_002", "prompt": "What is Big O notation used for?", "reference": "describing the time or space complexity of an algorithm"},
            {"id": "cs_003", "prompt": "What is the difference between TCP and UDP?", "reference": "TCP is connection-oriented and reliable while UDP is connectionless and faster but unreliable"},
            {"id": "cs_004", "prompt": "What does REST stand for in API design?", "reference": "Representational State Transfer"},
            {"id": "cs_005", "prompt": "What is a binary search tree?", "reference": "a tree data structure where each node has at most two children and left subtree values are less than the node while right subtree values are greater"},
            {"id": "cs_006", "prompt": "What is the purpose of a hash function?", "reference": "to map data of arbitrary size to fixed-size values"},
            {"id": "cs_007", "prompt": "What is recursion in programming?", "reference": "a function that calls itself to solve a problem by breaking it into smaller subproblems"},
            {"id": "cs_008", "prompt": "What is the difference between a stack and a queue?", "reference": "a stack is LIFO (last in first out) while a queue is FIFO (first in first out)"},
            {"id": "cs_009", "prompt": "What does SQL stand for?", "reference": "Structured Query Language"},
            {"id": "cs_010", "prompt": "What is a neural network?", "reference": "a machine learning model inspired by the brain consisting of layers of interconnected nodes that learn patterns from data"},
            # Reasoning
            {"id": "reas_001", "prompt": "If all cats are animals and all animals need food, do cats need food?", "reference": "yes"},
            {"id": "reas_002", "prompt": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "reference": "$0.05 or 5 cents"},
            {"id": "reas_003", "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?", "reference": "5 minutes"},
            {"id": "reas_004", "prompt": "A lily pad doubles in size every day. After 48 days it covers the lake. When did it cover half the lake?", "reference": "day 47"},
            {"id": "reas_005", "prompt": "There are 3 boxes labeled fruit, vegetables, and mixed. All labels are wrong. You can pick one item from one box. Which box do you pick from to correctly label all boxes?", "reference": "the box labeled mixed"},
            {"id": "reas_006", "prompt": "You have two ropes that each take exactly one hour to burn, but burn unevenly. How do you measure 45 minutes?", "reference": "light both ends of rope 1 and one end of rope 2 simultaneously; when rope 1 finishes (30 min) light the other end of rope 2"},
            {"id": "reas_007", "prompt": "If you overtake the person in second place in a race, what position are you in?", "reference": "second place"},
            {"id": "reas_008", "prompt": "A farmer has 17 sheep. All but 9 die. How many are left?", "reference": "9"},
            {"id": "reas_009", "prompt": "What comes once in a minute, twice in a moment, but never in a thousand years?", "reference": "the letter M"},
            {"id": "reas_010", "prompt": "John is taller than Mary. Mary is taller than Sue. Is John taller than Sue?", "reference": "yes"},
        ]

        return self._save("mmlu_sample", version, {
            "name": "mmlu_sample",
            "description": "50 MMLU-style questions across science, math, history, CS, and reasoning",
            "prompts": prompts,
        })

    def _create_reasoning_dataset(self, version: str) -> dict:
        prompts = [
            {"id": "r001", "prompt": "Explain step by step: If 2x + 3 = 11, what is x?", "reference": "x equals 4"},
            {"id": "r002", "prompt": "What is the next number in the sequence: 2, 4, 8, 16, ?", "reference": "32"},
            {"id": "r003", "prompt": "A train travels at 60mph for 2.5 hours. How far does it travel?", "reference": "150 miles"},
            {"id": "r004", "prompt": "If today is Wednesday and the meeting is in 10 days, what day is the meeting?", "reference": "Saturday"},
            {"id": "r005", "prompt": "Sort these numbers in ascending order: 7, 2, 9, 1, 5", "reference": "1, 2, 5, 7, 9"},
        ]
        return self._save("reasoning", version, {"name": "reasoning", "description": "Step-by-step reasoning tasks", "prompts": prompts})

    def _create_instruction_dataset(self, version: str) -> dict:
        prompts = [
            {"id": "i001", "prompt": "List exactly 3 programming languages in alphabetical order.", "reference": "Python, Rust, TypeScript"},
            {"id": "i002", "prompt": "Respond with only a single number: How many days in a week?", "reference": "7"},
            {"id": "i003", "prompt": "Write a one-sentence definition of machine learning.", "reference": "Machine learning is a field of artificial intelligence where systems learn patterns from data to make predictions or decisions."},
            {"id": "i004", "prompt": "Convert 100 Celsius to Fahrenheit. Give only the number.", "reference": "212"},
            {"id": "i005", "prompt": "Name the capital of France in one word.", "reference": "Paris"},
        ]
        return self._save("instruction_following", version, {"name": "instruction_following", "description": "Instruction adherence tasks", "prompts": prompts})
