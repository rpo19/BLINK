import re
import sys

# preprocess operations
# regex match + function to modify span

def preprocess_op_run(match, text, operation, mapping):
    new_span = mapSpan(match.span(), mapping)
    replacement, start_char, end_char = operation(match, text, new_span)
    if replacement is None:
        return text, mapping
    else:
        if start_char is None:
            start_char = new_span[0]
        if end_char is None:
            end_char = new_span[1]
        new_text = text[:start_char] + replacement + text[end_char:]

        # end_char --> start_char + len(replacement)
        # final char of match is now final char of replacement
        mapping[match.span()[1]-1] = start_char + len(replacement) - 1
        #print(mapping)
        
        mapping = sortMapping(mapping)
    
        return new_text, mapping

def getNext(candidates, matches):
    if candidates is None:
        candidates = []
        for i, m in enumerate(matches):
            try:
                candidates.append([next(m), i])
            except StopIteration:
                candidates.append(None)
    else:
        for i, c in enumerate(candidates):
            if c is None:
                try:
                    candidates[i] = [next(matches[i]), i]
                except StopIteration:
                    candidates[i] = None
    best = sorted(candidates, key=lambda x: x[0].span()[0] if x else sys.maxsize)[0]
    if best is not None:
        candidates[best[1]] = None
    return best, candidates

def getPrecedingMapping(index, mapping):
    keys = list(mapping.keys())
    kk = None
    for i, k in enumerate(keys):
        kk = k
        if k > index:
            # we are over, return previous
            if i == 0:
                # otherwise it will get last mapping with keys[-1]
                return None
            return keys[i-1]
    # return latest
    return kk

def mapSpan(span, mapping):
    start_i = getPrecedingMapping(span[0], mapping)
    start_shift = mapping[start_i] - start_i if start_i is not None else 0
    
    end_i = getPrecedingMapping(span[1], mapping)
    end_shift = mapping[end_i] - end_i if end_i is not None else 0
    
    if span[1] + end_shift - span[0] - start_shift == 0 and span[1] - span[0] != 0:
        end_shift += 1

    return span[0] + start_shift, span[1] + end_shift

def sortMapping(mapping):
    mapping_ = {k:v for k,v in mapping.items()} # do not modify original mapping
    def gen(mapping):
        for i, k in enumerate(mapping.keys()):
            yield k, i
    sorted_ = sorted(gen(mapping_), key=lambda x: x[0])
    #print(sorted_)
    delta = 0
    prevDelta = 0
    indexHistory = set([-1])
    for k, i in sorted_:
        mapping_[k] += delta
        if max(indexHistory) + 1 < i:
            # current mapping has been added after next
            # current delta - previous delta
            delta = mapping_[k] - k - prevDelta
        indexHistory.add(i)
        prevDelta = mapping_[k] - k
    return dict(sorted(mapping_.items()))

######################

# capitalize Names
def capitalize(match, text, new_span):
    # return replacement, start and end character of replacement or None
    replacement = ''
    for word in re.split(r'(\s+)', match.group(0)):
        replacement += word.capitalize() if word.isupper() else word
        
    return replacement, None, None
    
def quotation_marks(match, text, new_span):
    actual = text[new_span[0]:new_span[1]]
    if actual == "’" or actual == '`':
        return "'", None, None
    elif actual == "“" or actual == "”":
        return '"', None, None
    else:
        return None, None, None
    
def newlines(match, text, new_span):
    # check here because previous preprocessing could have been applied
    if re.match('^[\n\s]*\n[\n\s]*$', text[new_span[0]:new_span[1]]):
        return '\n', None, None
    else:
        return None, None, None
    
def add_space(match, text, new_span):
    # check here because previous preprocessing could have been applied
    actual = text[new_span[0]:new_span[1]].strip()
    if actual == '(':
        return "( ", None, None
    elif actual == ')':
        return " )", None, None
    elif actual == '"':
        return ' " ', None, None
    else:
        return None, None, None


# list containing (matching_regex, replace function)

preprocess_op = [
    ('[A-Z]{3,}', capitalize),
    ('[`“”]', quotation_marks),
    ('\s*([\(\)"])\s*', add_space),
    ('[\n\s]*\n[\n\s]*', newlines)
] 


# %%

def preprocess(text, preprocess_op=preprocess_op):
    text_ = text

    matches = []
    for expr,_ in preprocess_op:
        matches.append(re.finditer(expr, text))

    mapping = {}
    best, candidates = getNext(None, matches)
    #print(best)
    while best is not None:
        text_, mapping = preprocess_op_run(best[0], text_, preprocess_op[best[1]][1], mapping)
        best, candidates = getNext(candidates, matches)

    mapping_back = {v:k for k,v in mapping.items()}

    return text_, mapping, mapping_back