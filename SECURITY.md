# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for a vulnerability, leaked credential, or unsafe container or
dataset-handling behavior. Use GitHub's private vulnerability reporting for this repository when
available; otherwise contact the maintainers through the address listed on the REALM project page.

Include the affected commit or version, Docker image or SIF identity, reproduction steps, impact,
and any known mitigation. Do not include credentials, private dataset locations, cluster account
details, or other secrets in logs.

REALM executes third-party simulator code and consumes large external datasets and assets. Verify
their provenance and license terms, inspect image recipes before building, and never commit access
tokens or machine-specific environment files.
