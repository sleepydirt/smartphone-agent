FROM postgres:latest

ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=password
ENV POSTGRES_DB=smartphones_db

# Copy initialization files
COPY init.sql /docker-entrypoint-initdb.d/
COPY smartphone_inventory_sgd.csv /tmp/
COPY entrypoint.sh /docker-entrypoint-initdb.d/

# Fix line endings and make executable
RUN sed -i 's/\r$//' /docker-entrypoint-initdb.d/entrypoint.sh \
    && chmod +x /docker-entrypoint-initdb.d/entrypoint.sh

EXPOSE 5432