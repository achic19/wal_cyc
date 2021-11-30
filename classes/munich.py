my_cycle_count.to_postgis(name="cycle_count_data", con=engine, schema='production',
                          if_exists='replace',
                          dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})