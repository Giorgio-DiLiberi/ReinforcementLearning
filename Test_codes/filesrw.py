# code to learn and experiment how to read from files
with open("load.txt", "r") as input_file: # with open context
    input_file_all = input_file.readlines() # crate an array of strings containing all the file lines
    for line in input_file_all: # read line
        line = line.split() # splits lines into 2 strings
        globals()[line[0]] = line[1]

print(input_file_all)
print("cose= ", cose)
print(type(cose)) ## now "cose" vriable red from file are strings equal to strings

altro = float(cose) # to make them numbers typecasting is needed
print("altro = ", altro)
print(type(altro))