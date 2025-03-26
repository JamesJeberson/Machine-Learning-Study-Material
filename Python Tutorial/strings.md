### **Python String Functions, Methods, and Techniques**  

Python provides a rich set of built-in functions and methods to manipulate strings efficiently. Below is a **comprehensive list of functions, methods, and techniques** used with **Python strings**, along with **examples**.

---

## **1. String Creation & Basics**
```python
s = "Hello, World!"
s2 = 'Python is fun'
s3 = """Multiline 
String"""
print(s)  # Output: Hello, World!
```

---

## **2. String Indexing & Slicing**
```python
s = "Python"

# Indexing
print(s[0])   # Output: P
print(s[-1])  # Output: n

# Slicing
print(s[1:4])  # Output: yth
print(s[:4])   # Output: Pyth
print(s[2:])   # Output: thon
print(s[::2])  # Output: Pto  (skipping every 2nd character)
print(s[::-1]) # Output: nohtyP (reverse string)
```

---

## **3. String Built-in Functions**
| Function | Description |
|----------|-------------|
| `len(s)` | Returns length of string |
| `str(s)` | Converts any data type to string |
| `ord(c)` | Returns ASCII of a character |
| `chr(n)` | Returns character of ASCII value |
| `type(s)` | Returns type of variable |

```python
print(len("Python"))   # Output: 6
print(ord('A'))        # Output: 65
print(chr(97))         # Output: 'a'
print(str(123))        # Output: '123'
print(type("Hello"))   # Output: <class 'str'>
```

---

## **4. String Case Conversion Methods**
| Method | Description |
|--------|-------------|
| `s.upper()` | Converts to uppercase |
| `s.lower()` | Converts to lowercase |
| `s.title()` | Converts first letter of each word to uppercase |
| `s.capitalize()` | Capitalizes first letter |
| `s.swapcase()` | Swaps uppercase to lowercase and vice versa |

```python
s = "hello PYTHON"
print(s.upper())       # Output: HELLO PYTHON
print(s.lower())       # Output: hello python
print(s.title())       # Output: Hello Python
print(s.capitalize())  # Output: Hello python
print(s.swapcase())    # Output: HELLO python
```

---

## **5. String Searching Methods**
| Method | Description |
|--------|-------------|
| `s.find(sub)` | Returns index of first occurrence, `-1` if not found |
| `s.index(sub)` | Same as `find()` but raises error if not found |
| `s.rfind(sub)` | Returns last occurrence of substring |
| `s.startswith(prefix)` | Checks if string starts with prefix |
| `s.endswith(suffix)` | Checks if string ends with suffix |

```python
s = "Python is amazing"
print(s.find("is"))    # Output: 7
print(s.index("is"))   # Output: 7
print(s.rfind("is"))   # Output: 7
print(s.startswith("Py"))  # Output: True
print(s.endswith("ing"))   # Output: True
```

---

## **6. String Modification Methods**
| Method | Description |
|--------|-------------|
| `s.strip()` | Removes whitespace from start & end |
| `s.lstrip()` | Removes whitespace from start (left) |
| `s.rstrip()` | Removes whitespace from end (right) |
| `s.replace(old, new)` | Replaces substring |
| `s.ljust(width, fillchar)` | Left-aligns text, fills remaining space |
| `s.rjust(width, fillchar)` | Right-aligns text, fills remaining space |
| `s.center(width, fillchar)` | Centers text, fills remaining space |

```python
s = "  Python  "
print(s.strip())  # Output: "Python"
print(s.lstrip()) # Output: "Python  "
print(s.rstrip()) # Output: "  Python"
print(s.replace("Python", "Java"))  # Output: "  Java  "

s = "Hello"
print(s.center(10, '-'))  # Output: --Hello---
print(s.ljust(10, '-'))   # Output: Hello-----
print(s.rjust(10, '-'))   # Output: -----Hello
```

---

## **7. String Splitting & Joining**
| Method | Description |
|--------|-------------|
| `s.split(separator)` | Splits string into list |
| `s.rsplit(separator, maxsplit)` | Splits from right |
| `s.splitlines()` | Splits by new lines |
| `separator.join(iterable)` | Joins list into string |

```python
s = "Python,Java,C++"
print(s.split(","))  # Output: ['Python', 'Java', 'C++']

s = "Hello\nWorld"
print(s.splitlines())  # Output: ['Hello', 'World']

words = ['Python', 'Java', 'C++']
print(" - ".join(words))  # Output: "Python - Java - C++"
```

---

## **8. String Checking Methods**
| Method | Description |
|--------|-------------|
| `s.isalpha()` | Checks if all characters are alphabets |
| `s.isdigit()` | Checks if all characters are digits |
| `s.isalnum()` | Checks if all characters are alphanumeric |
| `s.isspace()` | Checks if string contains only spaces |
| `s.islower()` | Checks if all characters are lowercase |
| `s.isupper()` | Checks if all characters are uppercase |
| `s.istitle()` | Checks if string is title case |

```python
print("Hello".isalpha())   # Output: True
print("123".isdigit())     # Output: True
print("Hello123".isalnum())# Output: True
print("   ".isspace())     # Output: True
print("hello".islower())   # Output: True
print("HELLO".isupper())   # Output: True
print("Hello World".istitle())  # Output: True
```

---

## **9. String Formatting Methods**
| Method | Description |
|--------|-------------|
| `s.format()` | Formats string with placeholders `{}` |
| `f"{var}"` | f-string formatting |
| `% formatting` | Old style formatting |

```python
name = "John"
age = 25
print("My name is {} and I am {} years old".format(name, age))
# Output: My name is John and I am 25 years old

print(f"My name is {name} and I am {age} years old")
# Output: My name is John and I am 25 years old

print("My name is %s and I am %d years old" % (name, age))
# Output: My name is John and I am 25 years old
```

---

## **10. Reverse a String**
```python
s = "Python"
print(s[::-1])  # Output: nohtyP

# Using ''.join(reversed(s))
print(''.join(reversed(s)))  # Output: nohtyP
```

---

## **11. Remove Duplicates from a String**
```python
def remove_duplicates(s):
    return "".join(dict.fromkeys(s))  # Preserves order

print(remove_duplicates("programming"))  # Output: "progamin"
```

---

## **12. Count Occurrences of Each Character**
```python
from collections import Counter

s = "programming"
print(Counter(s))  # Output: Counter({'g': 2, 'r': 2, 'o': 1, 'p': 1, ...})
```

---
Here are some commonly asked **Python string coding interview questions** along with their **solutions**. 🚀  

---

### **1. Reverse a String**
**Problem:**  
Write a function to reverse a string without using built-in `reverse()` or slicing.  

**Solution:**
```python
def reverse_string(s):
    rev = ""
    for char in s:
        rev = char + rev
    return rev

print(reverse_string("hello"))  # Output: "olleh"
```
✅ **Alternative Method:**  
```python
def reverse_string(s):
    return s[::-1]

print(reverse_string("hello"))  # Output: "olleh"
```

---

### **2. Check if a String is a Palindrome**
**Problem:**  
A palindrome is a string that reads the same forward and backward.  
Write a function to check if a given string is a palindrome.  

**Solution:**
```python
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("madam"))   # Output: True
print(is_palindrome("hello"))   # Output: False
```
✅ **Alternative using Two Pointers:**  
```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

print(is_palindrome("racecar"))  # Output: True
```

---

### **3. Count the Number of Vowels in a String**
**Problem:**  
Write a function to count the number of vowels (`a, e, i, o, u`) in a string.  

**Solution:**
```python
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

print(count_vowels("hello world"))  # Output: 3
```

---

### **4. Remove Duplicates from a String**
**Problem:**  
Write a function to remove duplicate characters while maintaining order.  

**Solution:**
```python
def remove_duplicates(s):
    return "".join(dict.fromkeys(s))

print(remove_duplicates("banana"))  # Output: "ban"
```

---

### **5. Find the First Non-Repeating Character**
**Problem:**  
Given a string, find the first character that does not repeat.  

**Solution:**
```python
from collections import Counter

def first_non_repeating(s):
    freq = Counter(s)
    for char in s:
        if freq[char] == 1:
            return char
    return None  # If all characters repeat

print(first_non_repeating("swiss"))  # Output: "w"
```

---

### **6. Check if Two Strings are Anagrams**
**Problem:**  
Two strings are anagrams if they have the same characters in a different order.  

**Solution:**
```python
def is_anagram(s1, s2):
    return sorted(s1) == sorted(s2)

print(is_anagram("listen", "silent"))  # Output: True
print(is_anagram("hello", "world"))    # Output: False
```
✅ **Optimized Approach (Using Dictionary):**
```python
from collections import Counter

def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

print(is_anagram("race", "care"))  # Output: True
```

---

### **7. Find the Most Frequent Character in a String**
**Problem:**  
Find the character that appears the most in a given string.  

**Solution:**
```python
from collections import Counter

def most_frequent_char(s):
    freq = Counter(s)
    return max(freq, key=freq.get)

print(most_frequent_char("banana"))  # Output: "a"
```

---

### **8. Capitalize the First Letter of Each Word (Title Case)**
**Problem:**  
Convert a given string to title case (each word starts with an uppercase letter).  

**Solution:**
```python
def title_case(s):
    return " ".join(word.capitalize() for word in s.split())

print(title_case("hello world from python"))  # Output: "Hello World From Python"
```

---

### **9. Find the Longest Word in a Sentence**
**Problem:**  
Find the longest word in a given sentence.  

**Solution:**
```python
def longest_word(s):
    words = s.split()
    return max(words, key=len)

print(longest_word("I love programming in Python"))  # Output: "programming"
```

---

### **10. Remove All Whitespace from a String**
**Problem:**  
Remove all spaces (including new lines and tabs) from a string.  

**Solution:**
```python
def remove_whitespace(s):
    return "".join(s.split())

print(remove_whitespace("  Hello   World  "))  # Output: "HelloWorld"
```

---

### **11. Check if a String Contains Only Digits**
**Problem:**  
Write a function to check if a string consists only of digits (0-9).  

**Solution:**
```python
def is_numeric(s):
    return s.isdigit()

print(is_numeric("12345"))  # Output: True
print(is_numeric("12a45"))  # Output: False
```

---

### **12. Compress a String Using RLE (Run-Length Encoding)**
**Problem:**  
Compress a string by replacing consecutive duplicate characters with their count.  

**Example:** `"aaabbcddd"` → `"a3b2c1d3"`

**Solution:**
```python
def compress_string(s):
    if not s:
        return ""

    compressed = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(s[i - 1] + str(count))
            count = 1

    compressed.append(s[-1] + str(count))  # Append last character count
    return "".join(compressed)

print(compress_string("aaabbcddd"))  # Output: "a3b2c1d3"
```

---

### **13. Check if One String is a Rotation of Another**
**Problem:**  
Given two strings, check if one is a rotation of another.  

**Solution:**
```python
def is_rotation(s1, s2):
    return len(s1) == len(s2) and s1 in s2 + s2

print(is_rotation("waterbottle", "erbottlewat"))  # Output: True
print(is_rotation("hello", "lohel"))  # Output: True
print(is_rotation("hello", "world"))  # Output: False
```

---

### **14. Find All Permutations of a String**
**Problem:**  
Generate all possible permutations of a string.  

**Solution:**
```python
from itertools import permutations

def string_permutations(s):
    return ["".join(p) for p in permutations(s)]

print(string_permutations("abc"))
# Output: ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
```

---

### **15. Convert Roman Numerals to Integer**
**Problem:**  
Convert a Roman numeral string (e.g., `"XIV"`) to an integer (`14`).  

**Solution:**
```python
def roman_to_int(s):
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0

    for i in range(len(s)):
        if i > 0 and roman[s[i]] > roman[s[i - 1]]:
            total += roman[s[i]] - 2 * roman[s[i - 1]]
        else:
            total += roman[s[i]]

    return total

print(roman_to_int("XIV"))  # Output: 14
```

---
