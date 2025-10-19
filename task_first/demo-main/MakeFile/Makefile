#指定编译器
CC = gcc

#编译选项
CFLAGS = -g -O2 -Wall -Werror -I./include

# TODO
TARGET = hello.out
OBJS = hello.o add.o mu.o

# 默认目标
all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

# 编译主程序
hello.o: hello.c
	$(CC) $(CFLAGS) -c hello.c -o hello.o

# 编译 add.c
add.o: lib/add.c
	$(CC) $(CFLAGS) -c lib/add.c -o add.o

# 编译 mu.c
mu.o: lib/mu.c
	$(CC) $(CFLAGS) -c lib/mu.c -o mu.o

# 清理生成的文件
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean