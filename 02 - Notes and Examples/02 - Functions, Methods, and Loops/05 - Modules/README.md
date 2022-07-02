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

## Doctor Response Inputs

### Static Responses from the Doctor


```python
hedges = ("Please tell me more.",
          "Many of my patients tell me the same thing.",
          "Please continue.",
          "Let's put a pin in that one.",
          "Can you unpack that for me.")

```

### Dynamic responses from the Doctor


```python
qualifiers = ("Why do you say ",
              "You seem to think that ",
              "Can you explain why ")

```

### Pronouns to change from the patient's statement


```python
replacements = {"I":"you", "me":"you", "my":"your", "My":"your", 
                "we":"you", "us":"you", "mine":"yours"}

```

## Functions

### Function that Builds and returns a reply to an input sentence


```python

def reply(sentence):
  probability = random.randint(1, 4)
  if probability == 1:
    return random.choice(hedges)
  else:
    return random.choice(qualifiers) + changePersons(sentence)

```

### Function that Replaces first person pronouns with second person


```python
## Uses the dictionary replacements
def changePersons(sentence):
  words = sentence.split()
  replyWords = []
  for word in words:
      replyWords.append(replacements.get(word, word))
  return " ".join(replyWords)
```

## Main Method


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
    
    >> I am really mad
    Why do you say you am really mad
    
    >> Well, I hate life!
    Why do you say Well, you hate life!
    
    >> My dog is mean
    Can you explain why your dog is mean
    
    >> She doesn't ever listen!
    Many of my patients tell me the same thing.
    
    >> Are you kidding? You have not seen this animal in action though...
    You seem to think that Are you kidding? You have not seen this animal in action though...
    
    >> Yeah, tell me about it, bub
    Can you explain why Yeah, tell you about it, bub
    


```python
main()

```

    Good morning.  I hope you are well today?
    What can I do for you?
    
