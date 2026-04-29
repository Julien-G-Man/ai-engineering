scores = [92.3, 87.1, 95.8, 78.4]

first_number = scores[0]
second_number = scores[1]
last_number = scores[-1]

print("First number : ", first_number,
      "\nSecond number: ", second_number,
      "\nLast number  : ", last_number
      )

# modifying lists
scores.append(89.7)  # 
scores.insert(0, 90) # add 90 at index 0
print(scores)