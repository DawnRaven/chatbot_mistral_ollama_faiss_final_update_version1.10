# Khởi động dịch vụ nhanh không build lại
run:
	docker compose up -d

# Build lại toàn bộ và khởi động (chậm hơn)
build:
	docker compose up -d --build

# Dừng toàn bộ container
stop:
	docker compose down

# Xem log realtime của backend
logs:
	docker compose logs -f backend

# Restart backend
restart-backend:
	docker compose restart backend

# Xoá toàn bộ container, volume, image
clean:
	docker compose down --volumes --rmi all
