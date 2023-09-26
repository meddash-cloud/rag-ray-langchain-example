#!/bin/bash
set -a
source .env
psql $DB_CONNECTION_STRING -c "CREATE EXTENSION vector;"
