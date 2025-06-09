%python
def parse_copay(copay_str: str) -> int | None:
    import re
    nums = re.findall(r"\d+", copay_str)
    if not nums:
        return None
    return max(map(int, nums))


def query_plans(params: dict):
    """
    Query the plan_benefit_summary_view for the given service, location, and max_copay.

    params:
      - service: service description to filter (e.g., 'Diagnostic and Preventive Dental')
      - location: dict with keys 'city' and 'state'
      - max_copay: integer maximum copay value

    Returns a pandas DataFrame for plans matching criteria.
    """
    import pandas as pd
    from pyspark.sql import functions as F

    service = params['service']
    state = params['location'].get('state')
    max_copay = params.get('max_copay')

    df = spark.table("team5.team5.plan_benefit_summary_view") \
        .filter((F.col("category_name").like(f"%{service}%")) & (F.col("state") == state) & (F.col("network_description") == "In-Network")) \
        .select("plan_name") \
        .distinct() \
        .orderBy("plan_name") \
        .limit(5)

    display(df)


if __name__ == '__main__':
    params = {
        'service': 'Diagnostic and Preventive Dental',
        'location': {'city': 'San Francisco', 'state': 'California'},
        'max_copay': 50
    }
    query_plans(params)