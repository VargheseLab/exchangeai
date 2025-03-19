FROM imidocker.uni-muenster.de/published/exchangeai/exchangeai/optional_requirements:latest AS optional_requirements

FROM imidocker.uni-muenster.de/published/exchangeai/exchangeai/lite:latest
ENV FINETUNE_DISABLED=False

WORKDIR /app

COPY --from=optional_requirements /pip-packages /usr/local/lib/python3.11/dist-packages

EXPOSE 8000
CMD ["./startup.sh"]