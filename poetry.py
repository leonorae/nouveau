import sys
import datetime
import os.path
from textblob import TextBlob
import gpt_2_simple as gpt2
import json

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

poem_directory = './poems'

class Poem():
    def __init__(self, max_lines, generator):
        self.max_lines = max_lines
        self.lines = []
        self.generator = generator

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, line):
        return self.lines[line]
    
    def is_full(self):
        return len(self.lines) >= self.max_lines

    def add_line(self, line):
        if not self.is_full():
            self.lines.append(line)
        else:
            raise Exception("poem is full")
        
    def user_input_line(self):
        if not self.is_full():
            user = input()
            self.add_line(user)
            
    def generate(self):
        self.add_line(self.generator(self))

    def json(self):
        dict = {'generator': self.generator.__name__,
                'poem': self.lines}
                
        return json.dumps(dict)

def raw_gpt_generator(input):
    generated = gpt2.generate(sess,
                              length=10,
                              temperature=0.7,
                              prefix=input,
                              include_prefix=False,
                              return_as_list=True)[0]
    # TODO: trimming properly requires something like either tokenizing and rules or regex
    # +1 just because usually a space in generated
    trim = generated[len(input)+1:]
    return trim

def gpt_last(poem):
    return raw_gpt_generator(poem[-1])

def gpt_first(poem):
    return raw_gpt_generator(poem[0])

def gpt_closure(poem):
    if len(poem) == poem.max_lines-1:
        return gpt_first(poem)
    else:
        return gpt_last(poem)

def poem_loop(max_lines, generator):
    assert max_lines > 1

    poem = Poem(max_lines, generator)

    while not poem.is_full():
        poem.user_input_line()
        if not poem.is_full():
            poem.generate()
            print(poem[-1])
        else:
            break
        
    return poem

if __name__ == "__main__":
       
    lines = int(sys.argv[1])
    generator = globals()[sys.argv[2]]
    poem = poem_loop(lines, generator)

    datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.txt")
    filename = os.path.join(poem_directory, datetime)

    with open(filename, "w") as f:
        f.write(poem.json())

