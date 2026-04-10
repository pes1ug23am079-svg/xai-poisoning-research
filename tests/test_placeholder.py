"""
Starter tests for the XAI poisoning research project.
Add your own tests as the project grows.
"""


def test_placeholder():
    """Placeholder — replace with real tests as you build src/."""
    assert 1 + 1 == 2


# Example of what a real test will look like:
#
# from xai_poison.data.poisoner import inject_poison
#
# def test_poison_increases_label_noise():
#     clean_labels = [0, 1, 0, 1, 0]
#     poisoned_labels = inject_poison(clean_labels, rate=0.2)
#     noise = sum(a != b for a, b in zip(clean_labels, poisoned_labels))
#     assert noise > 0