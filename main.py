import pandas as pd

from src.preprocessing.encoder import Encoder


def test_onehot(df: pd.DataFrame):
    print("\n=== ONEHOT ===")

    enc = Encoder(df)
    enc.encode("onehot", columns=["color"])

    print(enc.get_data())


def test_label(df: pd.DataFrame):
    print("\n=== LABEL ===")

    enc = Encoder(df)
    enc.encode("label", columns=["color"])

    print(enc.get_data())


def test_ordinal(df: pd.DataFrame):
    print("\n=== ORDINAL ===")

    enc = Encoder(df)

    categories = {"size": ["S", "M", "L"]}

    enc.encode(
        "ordinal",
        columns=["size"],
        categories=categories,
    )

    print(enc.get_data())


def test_frequency(df: pd.DataFrame):
    print("\n=== FREQUENCY ===")

    enc = Encoder(df)
    enc.encode("frequency", columns=["city"])

    print(enc.get_data())


def test_target(df: pd.DataFrame):
    print("\n=== TARGET ===")

    enc = Encoder(df)

    enc.encode(
        "target",
        columns=["city"],
        target_column="target",
    )

    print(enc.get_data())


def test_transform(df: pd.DataFrame):
    print("\n=== TRANSFORM NEW DATA ===")

    enc = Encoder(df)
    enc.encode("label", columns=["color"])

    new_df = pd.DataFrame({"color": ["red", "blue", "green"]})

    transformed = enc.transform(new_df)

    print(transformed)


def test_inverse(df: pd.DataFrame):
    print("\n=== INVERSE TRANSFORM ===")

    enc = Encoder(df)

    enc.encode("label", columns=["color"])

    encoded = enc.get_data()

    decoded = enc.inverse_transform(encoded)

    print(decoded)


def main():

    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "blue"],
            "size": ["S", "M", "L", "S"],
            "city": ["Berlin", "Paris", "Berlin", "Rome"],
            "target": [1, 0, 1, 0],
        }
    )

    print("\nORIGINAL DATA")
    print(df)

    test_onehot(df)
    test_label(df)
    test_ordinal(df)
    test_frequency(df)
    test_target(df)
    test_transform(df)
    test_inverse(df)


if __name__ == "__main__":
    main()
