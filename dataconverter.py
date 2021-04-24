#conversion function for converting the df (includes the norm. difference of embeddings) into relative numbers:
def convert_absolute_to_relative(input):
    if input > 75: #>75: value 1
        output = 1
    elif input > 20:#20...75: value 0,5...1; calculate via y=mx+b
        output = 0.5/55*input+0.5-10/55
    elif input > 10:#10...20: value 0,1...0,5
        output = 0.04*input-0.3
    else: #0...10: value 0...0,1
        output = input/100
    return output

def convert_relative_to_class(rel_val):
    if rel_val == 1:
        output = "no match at all"
    elif rel_val >= 0.5:
        output = "almost no match"
    elif rel_val >= 0.1:
        output = "slightly matching"
    else:
        output = "match"
    return output
