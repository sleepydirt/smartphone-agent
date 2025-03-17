#!/bin/bash
set -e

# Wait until PostgreSQL is ready
until pg_isready -U admin; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

# Manually run the init.sql script on the default "postgres" database
echo "Running init.sql to create smartphones_db and smartphones table..."
psql -U admin -d postgres -f /docker-entrypoint-initdb.d/init.sql

# Wait until the smartphones table is created
until psql -U admin -d smartphones_db -tAc "SELECT to_regclass('public.smartphones');" | grep -q 'smartphones'; do
  echo "Waiting for 'smartphones' table to be created..."
  sleep 2
done

# Import CSV data
echo "Importing CSV data into smartphones table..."
psql -U admin -d smartphones_db -c "\COPY smartphones(id,brand,model,price,stock_status) FROM '/tmp/smartphone_inventory_sgd.csv' DELIMITER ',' CSV HEADER;"

exec docker-entrypoint.sh postgres
