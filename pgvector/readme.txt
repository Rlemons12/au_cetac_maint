- ONLY for PostgreSQL v17.3 - Windows x64
- Extract the zip file to your Postgres installed folder
- Run query: 
CREATE EXTENSION vector

- Run this query to check if the extension is enable (t):
SELECT extname,extrelocatable,extversion FROM pg_extension where extname='vector'