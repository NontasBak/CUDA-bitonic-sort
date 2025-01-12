CC=nvcc
TARGET=bitonic_sort_v0
SRC=bitonic_sort_v0.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

