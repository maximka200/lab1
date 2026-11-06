"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(character.lower() in "aeiou" for character in text)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((left - right) ** 2 for left, right in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        exponent = 0
        while number % divisor == 0:
            number //= divisor
            exponent += 1
        if exponent:
            factors.append(f"({divisor}**{exponent})" if exponent > 1 else f"({divisor})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    level = 0
    while total < cube_count:
        level += 1
        total += level * level
    return level if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1:]
    else:
        left, right = digits[:middle - 1], digits[middle + 1:]
    return sum(map(int, left)) == sum(map(int, right))
