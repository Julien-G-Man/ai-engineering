from IPython.display import display, Markdown, Latex, HTML

from main import get_completions

# Universal translator
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
]     
    
# Tone transformation
tone_transformation_prompt = f"""
Translate the following from slang to a business letter: 
'Dude, This is Joe, check out this spec on this standing lamp.'
"""

# Format Conversion
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

json_to_html_prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""

text = [ 
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for speling abilitty"  # spelling
]


proof_read_prompt = f"""
proofread and correct this review. Make it more compelling. 
Ensure it follows APA style guide and targets an advanced reader. 
Output in markdown format.
Text: ```{text}```
"""

if __name__ == "__main__":
    mode = ""
    if mode == "universal_translator":
        for issue in user_messages:
            prompt = f"Tell me what language this is: ```{issue}```"
            lang = get_completions(prompt)
            print(f"Original message ({lang}): {issue}")

            prompt = f"""
            Translate the following  text to English \
            and Korean: ```{issue}```
            """
            response = get_completions(prompt)
            print(response, "\n")
    
    elif mode == "json_to_html":
        response = get_completions(json_to_html_prompt)
        display(HTML(response))
      
    elif mode == "grammar_check":  
        for t in text:
            prompt = f"""Proofread and correct the following text
            and rewrite the corrected version. If you don't find
            and errors, just say "No errors found". Don't use 
            any punctuation around the text:
            ```{t}```"""
            response = get_completions(prompt)
            print(response)
    
    elif mode == "proof_read":
        response = get_completions(proof_read_prompt)
        display(Markdown(response))      
    else:
        response = get_completions(tone_transformation_prompt)
        print(response)
