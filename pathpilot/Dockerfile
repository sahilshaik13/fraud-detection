FROM python:3.12-slim

# ------------------------------------------------------------------
# Build tool-chain + headers for psycopg2-binary, NumPy, etc.
# libatlas-base-dev is gone in Debian Trixie ⇒ use BLAS/LAPACK refs
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
      build-essential gcc g++ \
            libpq-dev postgresql-client \
                  libblas-dev liblapack-dev \
                        && apt-get clean && rm -rf /var/lib/apt/lists/*

                        ENV PYTHONUNBUFFERED=1
                        WORKDIR /app

                        # Python deps
                        COPY requirements.txt .
                        RUN pip install --upgrade pip setuptools wheel
                        RUN pip install -r requirements.txt

                        # Project code
                        COPY . .

                        # Cloud Run listens on $PORT (default 8080)
                        CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--preload", "pathpilot.wsgi:application"]
                        