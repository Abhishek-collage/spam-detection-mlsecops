import pandera.pandas as pa
SCHEMA = pa.DataFrameSchema(
    {
        "Category":pa.Column(str, pa.Check.isin(["ham", "spam"]), nullable=False),
        "Message":pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=True
)

def validate(df):
    return SCHEMA.validate(df, lazy=True)

