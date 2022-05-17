## Modules
> Functions and other resources coded together  
> Many are always available but some must be explicitly loaded  

*Recommended Reading:  FoP – Chapter 2 (pages 54-60)*  

### Topics:
* Import conventions
* The Main Module


---

<br>

# Doctor AI Example

<a href="https://colab.research.google.com/github/BoomerPython/Week_2/blob/main/DSA_BoomerPython_Week2_DoctorProgram.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
## Sample notebook for exploring functions, lists, and dictionaries
## Developed from idea of ELIZA
## Code based on Lambert, 2019

```


```python
import random
```


```python
hedges = ("Please tell me more.",
          "Many of my patients tell me the same thing.",
          "Please continue.",
          "Let's put a pin in that one.",
          "Can you unpack that for me.")

```


```python
qualifiers = ("Why do you say ",
              "You seem to think that ",
              "Can you explain why ")

```


```python
replacements = {"I":"you", "me":"you", "my":"your", "My":"your", 
                "we":"you", "us":"you", "mine":"yours"}

```


```python
## Builds and returns a reply to an input sentence

def reply(sentence):
  probability = random.randint(1, 4)
  if probability == 1:
    return random.choice(hedges)
  else:
    return random.choice(qualifiers) + changePersons(sentence)

```


```python
## Replaces first person pronouns with second person
## Uses the dictionary replacements

def changePersons(sentence):
  words = sentence.split()
  replyWords = []
  for word in words:
      replyWords.append(replacements.get(word, word))
  return " ".join(replyWords)
```


```python
## Main function to handle the interaction between patient and doctor
print("Good morning.  I hope you are well today?")
print("What can I do for you?")

def main():
  print("Good morning.  I hope you are well today?")
  print("What can I do for you?")
while True:
  sentence = input("\n>> ")
  if sentence.upper() == "QUIT":
    print("Boomer Sooner!")
    break
  print(reply(sentence))
```

    Good morning.  I hope you are well today?
    What can I do for you?
    
    >> My mother and I do not get along
    Can you explain why your mother and you do not get along
    
    >> She always favors my sister
    Can you explain why She always favors your sister
    
    >> my dad and I get along fine
    Can you explain why your dad and you get along fine
    
    >> He helps me with my homework
    Please tell me more.
    
    >> When I need to study he asks me questions
    Why do you say When you need to study he asks you questions
    
    >> That is how he helps me with my homework
    Can you explain why That is how he helps you with your homework
    


```python
main()

```

    Good morning.  I hope you are well today?
    What can I do for you?
    
