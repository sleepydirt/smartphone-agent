FROM postgres:latest

ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=password
ENV POSTGRES_DB=smartphones_db

COPY ../db/init.sql /docker-entrypoint-initdb.d/
COPY ../db/smartphone_inventory_sgd.csv /tmp/smartphone_inventory_sgd.csv

COPY ../db/entrypoint.sh /docker-entrypoint-initdb.d/entrypoint.sh
RUN chmod +x /docker-entrypoint-initdb.d/entrypoint.sh

EXPOSE 5432