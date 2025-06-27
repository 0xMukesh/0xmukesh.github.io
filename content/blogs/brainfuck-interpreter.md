---
title: Writing a Brainfuck Interpreter
draft: false
date: 2024-12-15
---

I recently built a toy programming language ([brtlang](https://github.com/0xmukesh/brtlang)), which got me more interested about interpreters and compilers. So I thought of building a brainfuck interpreter (pretty obvious, didn't you see it coming?).

## Brainfuck

Brainfuck is an [esoteric programming language](https://en.wikipedia.org/wiki/Esoteric_programming_language) which is designed to be extremely minimalistic. The language consists of only eight simple commands, a data pointer, and an instruction pointer. Even though it has a very small set of keywords, it is a turning-complete language.

A brainfuck program is initialized with an array of 30,000 1 byte memory blocks (would be referring it as `tape`), in which each cell/block is initialized to zero, an instruction pointer that loops over the tape, and a data pointer that loops over the file contents.

The available 8 keywords/operators in brainfuck are:

1. `>` - move the instruction pointer to the next block
2. `<` - move the instruction pointer to the previous block
3. `+` - increase the value of current block
4. `-` - decrease the value of current block
5. `[` - loops until current block value is equal to 0
6. `]` - if current block value is non-zero, it jumps back to `[`. `[` and `]` combined form the loop
7. `,` - reads the character from `stdin` and stores it at the current block
8. `.` - prints out current block value to `stdout`

Any character other than the above mentioned 8 keywords must be considered as a comment, basically ignored by the interpreter.

### Hello World in Brainfuck

```
>>++++++++[<+++++++++>-]<.>++++[<+++++++>-]<+.+++++++..+++.>>++++++[<+++++++>-]<++.------------.>++++++[<+++++++++>-]<+.<.+++.------.--------.>>>++++[<++++++++>-]<+.
```

Let's break it down.

1. `>>++++++++[<+++++++++>-]<.`
2. `>++++[<+++++++>-]<+.`
3. `+++++++.`
4. `.`
5. `+++.`
6. `>>++++++[<+++++++>-]<++.`
7. `------------.`
8. `>++++++[<+++++++++>-]<+.`
9. `<.`
10. `+++.`
11. `------.`
12. `--------.`
13. `>>>++++[<++++++++>-]<+.`

Okay, now let's understand what each line does

1. `>>++++++++[<+++++++++>-]<.`

The instruction pointer moves one block to the right and increments the value by 8. The current state of tape - `[0,(8),...]` (`()` indicates that the pointer is currently at that block).

The program then enters a loop. The instruction pointer moves one block to the left and increments the value by 9. The current state of tape - `[(9),8,...]`. The instruction pointer moves one block to the right and decrements the value by 1. As the current block value is non-zero, it jumps back to `[` and it continues until value of the second block becomes zero, so 8 times. At the end of the loop, the value of first block would be equal to 72. The current state of tape - `[72,(0),...]`.

The instruction pointer moves one block to the left and prints out current block value. The current state of tape - `[(72),0,...]`. 72 is ASCII code for letter `H`.

2. `>++++[<+++++++>-]<+.`

The instruction pointer moves one block to the right and increments the value by 4. The current state of tape - `[72,(4),...]`.

The program enters a loop. The instruction pointer moves one block to the left and increments the value by 7. The current state of tape - `[(80),4,...]`.The instruction pointer moves one block to the right and decrements the value by 1. As the current block value is non-zero, it jumps back to `[` and it continues until the value of the second block becomes zero, so 4 times. At the end of the loop, the value of first loop would be equal to $72 + (4 \times 7) = 100$. The current state of tape - `[100,(0),...]`.

The instruction pointer moves one block to the left, increments the value by 1 and prints out current block's value. The current state of tape - `[(101),0,...]`. 101 is ASCII code for letter `e`.

3. `+++++++.`

The instruction pointer increments value of current block by 7 and prints out current block's value. The current state of tape - `[(108),0,...]`.

108 is ASCII code for letter `l`.

4. `.`

The instruction pointer prints out current block's value. The curent state of tape - `[(108),0,...]`.

108 is ascii code for letter `l`.

5. `+++.`

The instruction pointer increments value of current block by 3 and prints out current block's value. The current state of tape - [(111),0,...]

111 is ascii code for letter `o`.

6. `>>++++++[<+++++++>-]<++.`

The instruction pointer moves two blocks to the right and increments value of current block by 6. The current state of tape - `[111,0,(6),...]`.

The program enters a loop. The instruction pointer moves one block to the left and increments value of current block by 7. The current state of tape - `[111,(7),6,...]`

The instruction pointer moves one block to the right and decrements value of current block by 1. As value of current block is non-zero, the program jumps back to `[` and it continues until value of third block is equal to zero, so 6 times. The current state of tape - `[111,42,(0),...]`

The instruction pointer moves one block to the left, increments value of current block by 2 and prints out the value. The current state of tape - `[111,(44),0,...]`. 44 is ASCII code for `,`.

So yea, you get it. Let's now write the interpreter.

## Interpreter

Let's structure interpreter:

```go
type Interpreter struct {
    Tape  [30000]uint8
    Ptr   int
    Input []byte
}
```

1. `Tape` is 30,000 byte array which is initialized at the start of the program
2. `Ptr` is current index of instruction pointer
3. `Input` is the source code

`Tape` is of an array of type `uint8` to have wrapping

```go
func (p *Interpreter) Run() {
    bracketCounter := 0

    for i := 0; i < len(p.Input); i++ {
        switch p.Input[i] {
        case '>':
            p.Ptr++
        case '<':
            p.Ptr--
        case '+':
            p.Tape[p.Ptr]++
        case '-':
            p.Tape[p.Ptr]--
        case '.':
            fmt.Print(string(p.Tape[p.Ptr]))
        case ',':
            var input string
            if _, err := fmt.Scan(&input); err != nil {
                fmt.Println(InterpreterError("failed to read input", i))
                os.Exit(1)
            }
            runes := []rune(input)
            if len(runes) > 1 {
                fmt.Println(InterpreterError(fmt.Sprintf("recieved %s. expected single character input", input), i))
                os.Exit(1)
            }
            p.Tape[p.Ptr] = byte(runes[0])
        case '[':
            if p.Tape[p.Ptr] == 0 {
                bracketCounter++

                for p.Input[i] != ']' || bracketCounter != 0 {
                    i++

                    if p.Input[i] == '[' {
                        bracketCounter++
                    } else if p.Input[i] == ']' {
                        bracketCounter--
                    }
                }
            }
        case ']':
            if p.Tape[p.Ptr] != 0 {
                bracketCounter++

                for p.Input[i] != '[' || bracketCounter != 0 {
                    i--

                    if p.Input[i] == ']' {
                        bracketCounter++
                    } else if p.Input[i] == '[' {
                        bracketCounter--
                    }
                }
            }
        }
    }
}
```

I have defined `Run` method on `Interpreter` struct which is responsible for interpreting the source code.

The code might look overwhelming at first, so let's break down.

`Run` method loops the source code and checks value of each character. `i` in the `for` loop acts like data pointer, `p.Ptr` acts as instruction pointer and `p.Tape[p.Ptr]` acts as current block

1. If current character is `>` then `p.Ptr` is incremented by 1
2. If current character is `<` then `p.Ptr` is decremented by 1
3. If current character is `+` then value of current block (`p.Tape[p.Ptr]`) is incremented by 1
4. If current character is `-` then value of current block (`p.Tape[p.Ptr]`) is decremented by 1
5. If current character is `.` then string equivalent of current block value (`string(p.Tape[p.Ptr])`) is printed out
6. If current character is `,` then the input is read from `stdin`

   ```go
   var input string
   if _, err := fmt.Scan(&input); err != nil {
       fmt.Println(InterpreterError("failed to read input", i))
       os.Exit(1)
   }
   ```

   And checks whether the input is a _multi-character_ or not.

   ```go
   runes := []rune(input)
   if len(runes) > 1 {
       fmt.Println(InterpreterError(fmt.Sprintf("recieved %s. expected single character input", input), i))
       os.Exit(1)
   }
   ```

   `InterpreterError` is just an utility function which appends an prefix to an error message

   ```go
   func InterpreterError(msg string, idx int) string {
       return fmt.Sprintf("an error occured at %d character: %s", idx, msg)
   }
   ```

   If the input isn't a _multi-character_ then ASCII code of the first element is stored into the current block

   ```go
   p.Tape[p.Ptr] = byte(runes[0])
   ```

7. If current character is `[` then it checks current block value. If it is zero, then the loop is skipped until the matching `]`. If no, then it executes the loop body by moving the data pointer i.e. incrementing `i`

   `bracketCounter` is used to track how many pairs of brackets (`[` and `]`) the interpreter has encountered

   ```go
   if p.Tape[p.Ptr] == 0 {
       bracketCounter++

       for p.Input[i] != ']' || bracketCounter != 0 {
           i++

           if p.Input[i] == '[' {
               bracketCounter++
           } else if p.Input[i] == ']' {
               bracketCounter--
           }
       }
   }
   ```

8. If current character is `]` then it checks current block value. If it is zero, then the loop is exited. If no, then it jumps back to matching `[` to execute the loop body until current block value is equal to zero

   ```go
   if p.Tape[p.Ptr] != 0 {
       bracketCounter++

       for p.Input[i] != '[' || bracketCounter != 0 {
           i--

           if p.Input[i] == ']' {
               bracketCounter++
           } else if p.Input[i] == '[' {
               bracketCounter--
           }
       }
   }
   ```

And that's how you implement a brainfuck interpreter in under 100 lines in Golang.

[Source code](https://github.com/0xmukesh/brainfreeze)
