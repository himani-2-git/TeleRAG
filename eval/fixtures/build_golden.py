"""Generate eval/golden.json with exact verification against eval/fixtures/book.pdf."""

import json
from PyPDF2 import PdfReader

reader = PdfReader("eval/fixtures/book.pdf")
print("Extracting text from all pages...")
page_texts = [(idx, p.extract_text() or "") for idx, p in enumerate(reader.pages, 1)]
print(f"Extracted {len(page_texts)} pages.")

def find_matching_pages(terms):
    matches = set()
    for idx, txt in page_texts:
        if any(t.lower() in txt.lower() for t in terms):
            matches.add(idx)
    return sorted(matches)

def verify_terms_on_pages(terms, pages):
    for p in pages:
        txt = page_texts[p - 1][1]
        found = any(t.lower() in txt.lower() for t in terms)
        if not found:
            actual = find_matching_pages(terms)
            return False, f"None of {terms} found on physical page {p}. Term actually found on pages: {actual}"
    return True, "OK"

questions = [
    # Chapter 1: The way of the program (pp. 23-30)
    {
        "id": "q01",
        "question": "What is the difference between a high-level programming language like Python and a low-level language?",
        "expected_pages": [28],
        "category": "ch01_basics",
        "verify_terms": ["high-level", "low-level"]
    },
    {
        "id": "q02",
        "question": "What is the role of the Python interpreter and how does the prompt indicate it is ready for code?",
        "expected_pages": [24],
        "category": "ch01_basics",
        "verify_terms": ["interpreter", "prompt"]
    },
    {
        "id": "q03",
        "question": "What arithmetic operators are provided in Python and what does the ** operator compute?",
        "expected_pages": [25],
        "category": "ch01_basics",
        "verify_terms": ["arithmetic", "**", "exponentiation"]
    },
    {
        "id": "q04",
        "question": "What is the difference between syntax errors, runtime errors, and semantic errors in Python?",
        "expected_pages": [215],
        "category": "ch01_basics",
        "verify_terms": ["syntax error", "semantic error", "runtime error"]
    },

    # Chapter 2: Variables, expressions and statements (pp. 31-38)
    {
        "id": "q05",
        "question": "What happens when you use an illegal variable name such as a Python keyword like class or yield?",
        "expected_pages": [32],
        "category": "ch02_variables",
        "verify_terms": ["keywords", "syntaxerror"]
    },
    {
        "id": "q06",
        "question": "How does the modulus operator % work in Python and what does it compute?",
        "expected_pages": [61],
        "category": "ch05_conditionals",
        "verify_terms": ["modulus operator", "remainder"]
    },
    {
        "id": "q07",
        "question": "What is the order of operations followed by Python expressions (PEMDAS)?",
        "expected_pages": [34, 35],
        "category": "ch02_variables",
        "verify_terms": ["order of operations", "parentheses", "precedence"]
    },
    {
        "id": "q08",
        "question": "How does string concatenation using the + operator and string repetition using the * operator work in Python?",
        "expected_pages": [34],
        "category": "ch02_variables",
        "verify_terms": ["concatenation", "string operations"]
    },

    # Chapter 3: Functions (pp. 39-50)
    {
        "id": "q09",
        "question": "What built-in functions in Python convert values from one type to int, float, or str?",
        "expected_pages": [39],
        "category": "ch03_functions",
        "verify_terms": ["conversion", "int", "float"]
    },
    {
        "id": "q10",
        "question": "How do you import and access trigonometric or mathematical functions from the Python math module?",
        "expected_pages": [40, 41],
        "category": "ch03_functions",
        "verify_terms": ["math module", "math.sin", "radians"]
    },
    {
        "id": "q11",
        "question": "What is the difference between a parameter and an argument when defining and calling functions in Python?",
        "expected_pages": [43, 44],
        "category": "ch03_functions",
        "verify_terms": ["argument", "parameter"]
    },
    {
        "id": "q12",
        "question": "What is a stack diagram and how does it represent function calls and local variables?",
        "expected_pages": [45, 46],
        "category": "ch03_functions",
        "verify_terms": ["stack diagram", "frame"]
    },

    # Chapter 4: Case study: interface design (pp. 51-60)
    {
        "id": "q13",
        "question": "What is encapsulation in the context of writing reusable helper functions?",
        "expected_pages": [54],
        "category": "ch04_interface",
        "verify_terms": ["encapsulation"]
    },
    {
        "id": "q14",
        "question": "What is generalization and how does adding parameters make a function more general?",
        "expected_pages": [54],
        "category": "ch04_interface",
        "verify_terms": ["generalization"]
    },
    {
        "id": "q15",
        "question": "What is a docstring and how is it used to explain the interface and contract of a Python function?",
        "expected_pages": [58, 59],
        "category": "ch04_interface",
        "verify_terms": ["docstring", "triple quotes", "preconditions"]
    },

    # Chapter 5: Conditionals and recursion (pp. 61-72)
    {
        "id": "q16",
        "question": "How does floor division // differ from standard division / in Python 3?",
        "expected_pages": [61],
        "category": "ch05_conditionals",
        "verify_terms": ["floor division", "//"]
    },
    {
        "id": "q17",
        "question": "What are boolean expressions and the comparison operators ==, !=, >, <, >=, <= in Python?",
        "expected_pages": [62],
        "category": "ch05_conditionals",
        "verify_terms": ["boolean expression", "relational operators"]
    },
    {
        "id": "q18",
        "question": "What are chained conditionals using elif and when should they be used instead of nested conditionals?",
        "expected_pages": [63, 64],
        "category": "ch05_conditionals",
        "verify_terms": ["chained conditional", "elif", "nested"]
    },
    {
        "id": "q19",
        "question": "What happens if a recursive function has no base case or never reaches it in Python?",
        "expected_pages": [66, 67],
        "category": "ch05_conditionals",
        "verify_terms": ["infinite recursion", "RecursionError"]
    },
    {
        "id": "q20",
        "question": "How does the built-in input function prompt a user for keyboard input in Python?",
        "expected_pages": [67, 68],
        "category": "ch05_conditionals",
        "verify_terms": ["input(", "keyboard"]
    },

    # Chapter 6: Fruitful functions (pp. 73-84)
    {
        "id": "q21",
        "question": "What is a fruitful function and what does a function return if there is no return statement?",
        "expected_pages": [73, 74],
        "category": "ch06_fruitful",
        "verify_terms": ["fruitful function", "None"]
    },
    {
        "id": "q22",
        "question": "What is dead code in the context of statements following a return in a function?",
        "expected_pages": [74],
        "category": "ch06_fruitful",
        "verify_terms": ["dead code", "unreachable"]
    },
    {
        "id": "q23",
        "question": "What is incremental development and how does scaffolding help debug programs?",
        "expected_pages": [75, 76],
        "category": "ch06_fruitful",
        "verify_terms": ["incremental development", "scaffolding"]
    },
    {
        "id": "q24",
        "question": "What is a boolean function and how can it be used directly in conditional statements?",
        "expected_pages": [76, 77],
        "category": "ch06_fruitful",
        "verify_terms": ["boolean function", "is_divisible"]
    },
    {
        "id": "q25",
        "question": "How can you use isinstance to verify argument types and guard against invalid inputs in a factorial function?",
        "expected_pages": [81, 82],
        "category": "ch06_fruitful",
        "verify_terms": ["isinstance", "factorial", "guardian"]
    },

    # Chapter 7: Iteration (pp. 85-92)
    {
        "id": "q26",
        "question": "How does variable reassignment differ from mathematical equality in Python?",
        "expected_pages": [85, 86],
        "category": "ch07_iteration",
        "verify_terms": ["reassignment", "equality"]
    },
    {
        "id": "q27",
        "question": "How does the while statement implement indefinite iteration in Python?",
        "expected_pages": [86, 87],
        "category": "ch07_iteration",
        "verify_terms": ["while", "loop"]
    },
    {
        "id": "q28",
        "question": "How does the break statement allow you to exit from a loop prematurely?",
        "expected_pages": [88],
        "category": "ch07_iteration",
        "verify_terms": ["break"]
    },
    {
        "id": "q29",
        "question": "How does Newton's method compute square roots iteratively in Python?",
        "expected_pages": [88, 89],
        "category": "ch07_iteration",
        "verify_terms": ["newton's method", "square root", "formula"]
    },

    # Chapter 8: Strings (pp. 93-104)
    {
        "id": "q30",
        "question": "How do slice indices work on Python strings, and what does fruit[0:5] or fruit[:] return?",
        "expected_pages": [95],
        "category": "ch08_strings",
        "verify_terms": ["string slice", "slice"]
    },
    {
        "id": "q31",
        "question": "Why are Python strings immutable, and what error happens if you try to modify a character directly?",
        "expected_pages": [96],
        "category": "ch08_strings",
        "verify_terms": ["strings are immutable", "TypeError"]
    },
    {
        "id": "q32",
        "question": "How does the find method work on Python strings to locate substring positions?",
        "expected_pages": [97, 98],
        "category": "ch08_strings",
        "verify_terms": ["word.find", "find"]
    },
    {
        "id": "q33",
        "question": "How does the in operator check whether a character or substring appears inside a string in Python?",
        "expected_pages": [98],
        "category": "ch08_strings",
        "verify_terms": ["in operator", "boolean operator"]
    },

    # Chapter 10: Lists (pp. 111-124)
    {
        "id": "q34",
        "question": "What is the difference between strings and lists regarding mutability in Python?",
        "expected_pages": [112],
        "category": "ch10_lists",
        "verify_terms": ["lists are mutable", "bracket operator"]
    },
    {
        "id": "q35",
        "question": "What is the difference between the append and extend methods on a Python list?",
        "expected_pages": [114],
        "category": "ch10_lists",
        "verify_terms": ["append", "extend"]
    },
    {
        "id": "q36",
        "question": "How do map, filter, and reduce patterns operate on sequences in Python?",
        "expected_pages": [115, 116],
        "category": "ch10_lists",
        "verify_terms": ["reduce", "map", "filter"]
    },
    {
        "id": "q37",
        "question": "How do pop, del, and remove differ when removing elements from a list in Python?",
        "expected_pages": [116, 117],
        "category": "ch10_lists",
        "verify_terms": ["pop", "del", "remove"]
    },
    {
        "id": "q38",
        "question": "What is aliasing in Python lists and why can modifying an aliased list cause unexpected side effects?",
        "expected_pages": [119],
        "category": "ch10_lists",
        "verify_terms": ["aliasing", "reference", "identical"]
    },

    # Chapter 11: Dictionaries (pp. 125-136)
    {
        "id": "q39",
        "question": "How does a Python dictionary implement key-value pairs and what is the in operator checking on dictionaries?",
        "expected_pages": [125, 126],
        "category": "ch11_dictionaries",
        "verify_terms": ["mapping", "keys", "values"]
    },
    {
        "id": "q40",
        "question": "How can a dictionary be used as a collection of counters with the get method in Python?",
        "expected_pages": [127, 128],
        "category": "ch11_dictionaries",
        "verify_terms": ["histogram", "get method", "counter"]
    },
    {
        "id": "q41",
        "question": "What is memoization and how can a dictionary memo speed up recursive calculations like Fibonacci?",
        "expected_pages": [131, 132],
        "category": "ch11_dictionaries",
        "verify_terms": ["memo", "memoization", "fibonacci"]
    },

    # Chapter 12: Tuples (pp. 137-146)
    {
        "id": "q42",
        "question": "What is tuple assignment in Python and how does it allow swapping two variables without a temporary variable?",
        "expected_pages": [138, 139],
        "category": "ch12_tuples",
        "verify_terms": ["tuple assignment", "swap"]
    },
    {
        "id": "q43",
        "question": "How does the gather parameter *args work to accept a variable number of arguments into a tuple?",
        "expected_pages": [140],
        "category": "ch12_tuples",
        "verify_terms": ["gather", "args", "scatter"]
    },
    {
        "id": "q44",
        "question": "How does the zip function combine two or more sequences into a list or iterator of tuples?",
        "expected_pages": [141, 142],
        "category": "ch12_tuples",
        "verify_terms": ["zip", "iterator"]
    },

    # Chapter 14: Files (pp. 159-168)
    {
        "id": "q45",
        "question": "How does the dbm module work in Python to create and store persistent key-value databases on disk?",
        "expected_pages": [163],
        "category": "ch14_files",
        "verify_terms": ["dbm", "database"]
    },
    {
        "id": "q46",
        "question": "What does the pickle module do to serialize and deserialize arbitrary Python objects into strings or bytes?",
        "expected_pages": [164],
        "category": "ch14_files",
        "verify_terms": ["pickle", "pickling", "dumps"]
    },

    # Chapter 15: Classes and objects (pp. 169-176)
    {
        "id": "q47",
        "question": "What is a programmer-defined type or class in Python and how do you instantiate an object from it?",
        "expected_pages": [169, 170],
        "category": "ch15_classes",
        "verify_terms": ["programmer-defined", "class Point", "instantiation"]
    },
    {
        "id": "q48",
        "question": "What is the difference between shallow copy and deep copy when copying objects with the copy module?",
        "expected_pages": [174, 175],
        "category": "ch15_classes",
        "verify_terms": ["copy.copy", "deepcopy", "shallow"]
    },

    # Chapter 16 & 17: Classes, functions, and methods (pp. 177-192)
    {
        "id": "q49",
        "question": "What is the difference between a pure function and a modifier method on an object?",
        "expected_pages": [178, 179],
        "category": "ch16_functions_methods",
        "verify_terms": ["pure function", "modifier"]
    },
    {
        "id": "q50",
        "question": "What is polymorphism in Python and how does it allow functions to work with arguments of different types?",
        "expected_pages": [189, 190],
        "category": "ch17_methods",
        "verify_terms": ["polymorphism", "polymorphic"]
    },
]

# 10 Unanswerable queries (off-topic)
unanswerable = [
    {
        "id": "u01",
        "question": "What is the recommended recipe and starter feeding ratio for baking sourdough bread?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u02",
        "question": "How do quantum computers use superconducting qubits and superposition for quantum supremacy?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u03",
        "question": "What are the common symptoms of a blown head gasket in an automobile combustion engine?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u04",
        "question": "How do you prune panicle hydrangeas in late winter to encourage optimal summer blooming?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u05",
        "question": "What were the major military campaigns of Hannibal Barca during the Second Punic War?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u06",
        "question": "How do high-bypass turbofan jet engines compress air and generate thrust in commercial aviation?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u07",
        "question": "What is the standard procedure for pre-infusion and brewing espresso using an E61 group head?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u08",
        "question": "How do central banks adjust monetary policy using repo rates to target headline inflation?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u09",
        "question": "What soil pH and organic fertilizer requirements are needed to grow highbush blueberries?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    },
    {
        "id": "u10",
        "question": "How do lithium-ion battery management systems perform passive cell balancing during high-voltage charging?",
        "expected_pages": [],
        "answerable": False,
        "category": "unanswerable"
    }
]

print(f"Validating {len(questions)} answerable questions against book.pdf...")
all_valid = True
for q in questions:
    ok, msg = verify_terms_on_pages(q["verify_terms"], q["expected_pages"])
    if not ok:
        print(f"FAIL: {q['id']} - {msg}")
        all_valid = False
        
if all_valid:
    print("All 50 answerable questions verified against physical pages in book.pdf!")
    # Build clean output
    clean_dataset = []
    for q in questions:
        clean_dataset.append({
            "id": q["id"],
            "question": q["question"],
            "expected_pages": q["expected_pages"],
            "answerable": True,
            "category": q["category"]
        })
    for u in unanswerable:
        clean_dataset.append(u)
        
    with open("eval/golden.json", "w", encoding="utf-8") as f:
        json.dump(clean_dataset, f, indent=2)
    print("Wrote 60 items (50 answerable + 10 unanswerable) to eval/golden.json")
else:
    print("Verification failed! Fix page numbers.")
