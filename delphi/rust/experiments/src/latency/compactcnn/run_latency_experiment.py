import os

def main():
    output_file = "list.txt"
    os.system("ls > {}".format(output_file))

if __name__ == "__main__":
    main()