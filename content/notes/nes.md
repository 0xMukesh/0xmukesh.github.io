---
title: NES overview
draft: false
date: 2026-03-06
---

i recently started working on a NES emulator ([veyra](https://github.com/0xmukesh/veyra)). this led me to explore multiple resources to understand how the NES works under the hood. the following content serves as an aggregated view of those resources. all referenced materials are linked at the end.

# CPU

NES used a variant of [standard 6502](https://en.wikipedia.org/wiki/MOS_Technology_6502) processor, it was [ricoh 2a03](https://en.wikipedia.org/wiki/Ricoh_2A03). it is a 8-bit processor and follows little endian format i.e. least significant byte is stored at 
the lowest memory address. some of the major difference between standard 6502 and ricoh 2a03 is that the latter could serve as a pseudo APU (audio processing unit) and also a CPU and it didn't support [BCD](https://en.wikipedia.org/wiki/Binary-coded_decimal) (binary coded decimal)

CPU accesses memory through buses. the memory within NES can be split up into 3 major parts:
- cartridge ROM
- internal RAM of CPU
- I/O registers -- NES uses memory mapped I/O i.e. data can be transferred among devices by writing to a particular address in memory

![](../assets/nes/processor-diagram.png)

the above diagram shows that the processor system has 3 buses via which different components communicate with each other. here is what each one of them do in brief:
- 8-bit data bus -- carries the data which is being read or written. it is bidirectional in nature
- 16-bit address bus -- carries the 16-bit address where the CPU is trying to read/write to
- 8-bit control bus -- carries signals like `READ`, `WRITE`, `RESET`, `IRQ` (interrupt requests). it is used to know what _kind_ of operation is being executed

as NES used a 16-bit address, it could support 64 KB of memory with addresses from 0x0000 to 0xFFFF. the following image shows the memory map of the CPU, which shows how the 64 KB is split and how various information is laid out in the memory.


## memory map

![](../assets/nes/cpu-memory-map.png)

### RAM

**0x0000 to 0x1FFF** contains CPU's RAM. 0x2000 in decimal in 8192, which would translate to 8 KiB, but an interesting part regarding this is that, NES has only 2 KiB. the usable RAM is present from 0x0000 to 0x01FF, which is then _mirrored_ 3 times to create 8 KB RAM in memory map. the reason being in this is due to **mirroring**. if 0x0000 and 0x1FFF is represented in binary:
  - 0x0000 = 0000 0000 0000 0000 
  - 0x1FFF = 0001 1111 1111 1111
  
the address decoder logic to know whether the RAM was being accessed was just a simple 3-input NOR gate to check whether upper 3-bits (A15, A14 and A13) of the address are all zero or not. this leaves with more 13 bits to work with and 2^13 = 8 KiB. the CPU has 2 KiB internal RAM because A12 and A11 are ignored. if **0x0000 to 0x0800** was chosen for RAM in memory map then address decoder logic to know whether RAM was being accessed would require to check whether A15, A14, A13, A12 and A11 are all zero. at the time when NES was being developed only 3-input and 4-input NOR gate. due to technology constraints and to keep the product's cost low, A12 and A11 are just ignored which causes mirroring.
<br/>
<br/>
RAM memory map is divided into multiple sub-regions as follows:
- **0x0000 to 0x00FF** refers to zero page as they represent in first 256 bytes and some instructions use zero page addressing modes for quicker read and writes.
- **0x0100 to 0x01FF** contains the stack of the CPU 
- **0x0200 to 0x07FF** contains the actual RAM, which is of 1.5 KiB
- **0x0800 to 0x1FFF** contains the mirrors of **0x0000 to 0x07FF** 

### I/O registers

**0x2000 to 0x401F** contains memory mapped I/O registers. PPU (picture processing unit, more on this in later sections) related registers are present in **0x2000 to 0x2008** and a similar mirroring effect can be since here as well. remaining I/O registers are present in **0x4000 to 0x4020**.

### expansion ROM

**0x4020 to 0x5FFF** contains cartridge hardware extensions such as additional program ROM, extra sound hardware (extra sound channels), custom registers etc.

### SRAM

**0x6000 to 0x7FFF** refers to SRAM which was generally used to save game state.

### PRGROM

**0x8000 to 0xFFFF** contains the game's actual code i.e. raw CPU instructions which would iterated and executed by CPU each step. this region is memory-mapped with cartridge's ROM. it is split up into two 16 KiB banks (similar wording would be seen in iNES file format). if there is a game whose code is more 32 KiB in size, then MMC comes into picture. MMC or memory management controller swaps different chunks of ROM into CPU's addressable window on the fly, can determine which banks is to be loaded into the memory. it sits in between the CPU and ROM, intercepting the signals and remapping them. 

## registers

6502 has 3 special purpose registers (program counter, stack pointer, processor status) and 3 general purpose registers (accumulator, X register, Y register). 

- **program counter (PC)** -- it holds the 16-bit address of the the next instruction which is to be executed.
- **stack pointer (SP)** -- it holds a 8-bit value which acts like a offset from 0x0100. the initial value is 0x01FF and when a byte is pushed to the stack, stack pointer is decreased and vice versa.
- **accumulator (A)** -- it is a 8-bit register which stores the value of arithemtic and logic operations.
- **index register X/Y (X/Y)** -- these are 8-bit registers which are typically used as counters or an offset for certain addressing modes. 
- **processor status (P)** -- it is a 8-bit register which contains the 8 single bit status flags
    ![](../assets/nes/processor-status.png)
    
    - **bit 7 (negative, N)** -- set if the result of an operation is negative i.e. bit 7 of the result is 1
    - **bit 6 (overflow, V)** -- set if result of a signed arithmetic operation overflowed i.e. result is too large for a signed byte. it is determined by looking at the looking at the carry between bits 6 and 7 and between bit 7 and the carry flag i.e. whether the carry in bit and carry out bit for bit 7 are different or not
    - **bit 5 (unused)** -- always set to 1
    - **bit 4 (break, B)** -- it acts like a transient signal in the CPU. if the flags were pushed while processing an interrupt, then it is 0 and 1 when it is pushed by instructions (`BRK`, `PHP`)
    - **bit 3 (decimal mode, D)** -- when decimal mode flag is set, processor will obey the rules of [binary coded decimal](https://en.wikipedia.org/wiki/Binary-coded_decimal) during arithmetic operations. it isn't used in ricoh 2a03 as it doesn't support binary-coded decimal.
    - **bit 2 (interrupt disable, I)** -- set when [SEI](https://www.nesdev.org/obelisk-6502-guide/reference.html#SEI) instruction is executed
    - **bit 1 (zero, Z)** -- set if the result of the operation was 0
    - **bit 0 (carry, C)** -- set if an operation produced a carry/borrow

## addressing modes

6502 provides multiple different ways on how to address particular locations present in the memory.

- **zero page** -- zero page addressing takes a single operand which acts like a pointer to an address in zero page where the data to be operated on can be found.
- **indexed zero page** -- indexed zero page addressing takes a single operand and adds the value of index register (X register for `zero page, X` and Y register for `zero page, Y`) to it to give an address in (0x0000 to 0x00FF) i.e. addition is wrapped around 
- **absolute** -- absolute addressing takes two single operand which combined forms full 16-bit address which contains the data to be operated on. the sequence of operands is little endian format i.e. least significant byte first.
- **indexed absolute** -- similar to indexed zero page, indexed absolute takes two operands and the address is then added with the value of index registers to the final address.
- **indirect** -- indirect addressing takes two operands which combined forms a 16-bit address which stores the least significant byte of another 16-bit address where the data to be operated on is stored.
- **implied** -- instructions which don't require access to operands to be stored in memory.
- **accumulator** -- instructions which directly operate on accumulator register use this addressing mode.
- **immediate** -- immediate addressing mode takes a single operand which is a 8-bit constant.
- **relative** -- relative addressing mode takes a single operand which is a signed 8-bit constant which is used to update value of program counter if a certain condition is met. the condition is dependant on the instruction and the program counter increments by 2 (one of the opcode and another one for the operand) regardless of whether the condition is met or not.
- **indexed indirect** -- indexed indirect (or) pre-indexed addressing mode takes a single operand and then adds it to X register (with wrap around) to give address of the least significant byte of the target 16-bit address.
- **indirect indexed** -- indirect indexed (or) post-indexed addressing mode takes a single operand which gives zero page address of least significant byte of another 16-bit address which is then added to Y register to give the target address.

# PPU

# resources

- https://www.nesdev.org/NESDoc.pdf
- https://www.nesdev.org/obelisk-6502-guide/
- https://bugzmanov.github.io/nes_ebook/
