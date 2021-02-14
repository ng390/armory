"""
This module enables loading of different perturbation functions in poisoning
"""

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning import perturbations
from armory.art_experimental.attacks.poison_square_bd import add_large_pattern_bd


def poison_loader_GTSRB(**kwargs):
    poison_type = kwargs["poison_type"]
    if poison_type == "square":
        square_size = kwargs.get("square_size")
        if square_size is None:
            raise ValueError("poison type 'square' requires 'size' kwarg")
        distance = kwargs.get("distance")
        if distance is None:
            raise ValueError("poison type 'square' requires 'distance' kwarg")

        def mod(x):
            return add_large_pattern_bd(x, distance, square_size, pixel_value=1)

    elif poison_type == "pattern":

        def mod(x):
            return perturbations.add_pattern_bd(x, pixel_value=1)

    elif poison_type == "pixel":

        def mod(x):
            return perturbations.add_single_bd(x, pixel_value=1)

    elif poison_type == "image":
        backdoor_path = kwargs.get("backdoor_path")
        if backdoor_path is None:
            raise ValueError(
                "poison_type 'image' requires 'backdoor_path' kwarg path to image"
            )
        size = kwargs.get("size")
        if size is None:
            raise ValueError("poison_type 'image' requires 'size' kwarg tuple")
        size = tuple(size)

        def mod(x):
            return perturbations.insert_image(x, backdoor_path=backdoor_path, size=size)

    else:
        raise ValueError(f"Unknown poison_type {poison_type}")

    return PoisoningAttackBackdoor(mod)
