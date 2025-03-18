CREATE DATABASE smartphones_db;
\c smartphones_db;
CREATE TABLE IF NOT EXISTS smartphones (
    id SERIAL PRIMARY KEY,
    brand TEXT,
    model TEXT,
    price INTEGER,
    stock_status TEXT
);