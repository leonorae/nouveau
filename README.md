# nouveau
GPT-2 Interactive Poetry Generation

Instructions for use:
1. Download the model here, or run data.py and follow the instructions in the colab notebook linked in that file

2. run ```python3 poetry.py <number_lines> <generator>

3. write a poem, the program will terminate when max number of lines specified is reached.
   
Current generators are:
gpt_last: generate based on the last line input by the user
gpt_closure: like above but generate based on first line for final line
