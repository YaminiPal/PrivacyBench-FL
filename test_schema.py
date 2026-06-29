from preprocessing.schema_check import SchemaChecker

checker = SchemaChecker(target_column="income")

df, schema = checker.run("data/raw/adult.csv")