from typing import Dict, Union, Any, List

import dataclasses
import pandas as pd
from fastai.tabular import transform
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing

DataFrame = pd.DataFrame


@dataclasses.dataclass
class QuantileNormalize(transform.TabularProc):
    quantile_transformers: Union[Dict[str, preprocessing.QuantileTransformer], Any] = None

    def apply_train(self, df: DataFrame):
        self.quantile_transformers = {}
        for var in self.cont_names:
            vals = df[var].values.reshape(-1, 1)
            assert is_numeric_dtype(vals), (f"""Cannot normalize {var} column as it isn't numerical.
                Are you sure it doesn't belong in the categorical set of columns?""")
            self.quantile_transformers[var] = preprocessing.QuantileTransformer(output_distribution='uniform').fit(vals)
            df[var] = self.quantile_transformers[var].transform(vals)

    def apply_test(self, df: DataFrame):
        for n in self.cont_names:
            vals = df[n].values.reshape(-1, 1)
            df[n] = self.quantile_transformers[n].transform(vals)


@dataclasses.dataclass
class Power(transform.TabularProc):
    """
    For each numeric column, crete two additional columns that takes the square and the square
    root of that column.
    """
    power_cols: List[str] = dataclasses.field(default_factory=list)

    def apply_train(self, df: DataFrame):
        new_cols = []
        for var in self.cont_names:
            vals = df[var].values
            assert is_numeric_dtype(vals), (f"""Cannot normalize {var} column as it isn't numerical.
                            Are you sure it doesn't belong in the categorical set of columns?""")
            df[var + '_square'] = vals ** 2
            df[var + '_sqrt'] = vals ** 0.5

            self.power_cols.append(var)
            new_cols.append(var + '_square')
            new_cols.append(var + '_sqrt')

        self.cont_names.extend(new_cols)

    def apply_test(self, df: DataFrame):
        new_cols = []
        for var in self.cont_names:
            if var in self.power_cols:
                vals = df[var].values
                df[var + '_square'] = vals ** 2
                df[var + '_sqrt'] = vals ** 0.5

                new_cols.append(var + '_square')
                new_cols.append(var + '_sqrt')

        for new_col in new_cols:
            if new_col not in self.cont_names:
                self.cont_names.append(new_col)
