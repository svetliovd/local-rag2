# Стартира приложението
up:
	docker compose --profile gpu up
	# Изчаква се малко, за да се увери, че ollama контейнерът е напълно стартиран
	sleep 20
	docker exec local-rag-ollama-1 ollama pull todorov/bggpt:9B-IT-v1.0.Q8_0

# Стартира приложение в "откачен" режим, т.е. работата му не зависи от сесията на терминала
up-d:
	docker compose --profile gpu up -d
	# Изчаква се малко, за да се увери, че ollama контейнерът е напълно стартиран
	sleep 20
	docker exec local-rag-ollama-1 ollama pull todorov/bggpt:9B-IT-v1.0.Q8_0

# Стартиране със зареждане наново на всички зависимости и модули
build:
	docker compose --profile gpu up --build
	# Изчаква се малко, за да се увери, че ollama контейнерът е напълно стартиран
	sleep 20
	docker exec local-rag-ollama-1 ollama pull todorov/bggpt:9B-IT-v1.0.Q8_0

# Спиране и премахване на контейнерите
down:
	docker compose down

# Спиране, премахване на контейнерите и изчистване на томовете. Внимание! Загуба на информация, ако има такава!
down-v:
	docker compose down -v

# Проверка на доклади
logs:
	docker compose logs -f

# Проверка на статус на контейнерите
ps:
	docker compose ps

# Презареди само дисковите образи
rebuild:
	docker compose build

