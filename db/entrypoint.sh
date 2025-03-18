set -e

until pg_isready -U admin; do
  sleep 2
done

psql -U admin -d postgres -f /docker-entrypoint-initdb.d/init.sql

until psql -U admin -d smartphones_db -tAc "SELECT to_regclass('public.smartphones');" | grep -q 'smartphones'; do
  sleep 2
done

psql -U admin -d smartphones_db -c "\COPY smartphones(id,brand,model,price,stock_status) FROM '/tmp/smartphone_inventory_sgd.csv' DELIMITER ',' CSV HEADER;"

exec docker-entrypoint.sh postgres