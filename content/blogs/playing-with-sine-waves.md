---
title: Playing with sine waves
draft: false
date: 2024-12-28
---

While scrolling through YouTube, I came across this video on my feed by [javidx9](https://www.youtube.com/@javidx9) - [https://youtu.be/tgamhuQnOkM?si=QnRopBay40F54sAH](https://youtu.be/tgamhuQnOkM?si=QnRopBay40F54sAH) about building a sound synthesizer in C++, so I thought of giving audio programming a shot. In this blog post, we'll cover a few of the basic topics and build a simple program which would generate a sine wave of a certain frequency and save it to a `.wave` file, amplify it and perform stereopanning on a given `.wave` file.

## What is sound?

Sound is a phenomenon caused by vibration in particles that propagates as a wave through a transmission medium such as air, water, or solids. Most of the sounds in the real world propagate in the form of a sine wave (or combination of different sine waves).

For example, the note A above middle C on the piano propagates as a (almost) pure sine wave with a frequency of 440Hz ([ref: how can a piano key only have one frequency?](https://www.reddit.com/r/AskPhysics/comments/10tffnh/how_can_a_piano_key_have_only_one_frequency/)) and it can be mathematically represented as follows:

> $y = \sin(880\pi x)$

## What is audio?

Sound is a mechanical wave energy, while audio is the electrical representation of that sound wave.

A microphone converts the mechanical sound waves into analog signals which is later passed through an analog-digital converter (ADC) which converts these analog signals into digital signals, which would be understood by a computer.

There are two important keywords in the context of digital audio - **bit rate** and **sample frequency**

The analog signals hold information about various wave characteristics at that particular instant such as the amplitude. Sample frequency is the number of times a _snapshot_ of these characteristics is taken and these snapshots are later used to re-create the sound wave.

Most of the audio which is delivered nowadays either uses 44.1 kHz or 48 kHz as the sample frequency. The frequency limit of humans range from 20 Hz to 22 kHz. Nyquist rate is the minimum sampling rate needed to accurately represent a signal and it is twice the highest frequency of the signal.

> $44100 = 2 \times 22000 + 100$

Why isn't it 44 kHz? Well, an additional 100 Hz sorta acts like a transition band or room for error, which prevents unwanted distortion in the higher frequencies.

In the early days, digital audio was stored on modified video recorders and 44.1 kHz worked perfectly with the video equipment at that time and it became the industry standard.

Bit depth is related to the precision of each _snapshot_. If the bit depth is 16 then the maximum number which could be represented is +32767 (((2^16 - 1) - 1)/2) and the least is -32767. A snapshot could either have +ve or -ve amplitude. So in simple words - the higher the bit depth, the more clearly it is represented digitally, at the regions with really high or really low frequencies.

## Crafting initial sounds

Let's create a simple program which would generate a sine wave of 440 Hz.

```go
package main

import (
  "encoding/binary"
  "fmt"
  "math"
  "os"
  "time"
)

const (
  duration   = 5
  sampleRate = 44100
  freq       = 440
)

func main() {
  ns := duration * sampleRate
  angle := (math.Pi * 2.0) / float64(ns)

  f, err := os.Create("wave.bin")
  if err != nil {
    panic(err.Error())
  }
  start := time.Now()

  for i := 0; i < ns; i++ {
    sample := math.Sin(angle * freq * float64(i))
    var buf [4]byte

    binary.LittleEndian.PutUint32(buf[:], math.Float32bits(float32(sample)))

    if _, err := f.Write(buf[:]); err != nil {
      panic(err.Error())
    }
  }

  fmt.Printf("done - %dms\n", time.Since(start).Milliseconds())
}
```

The above program generates a `.bin` file containing binary representation of the audio samples, at a sample rate of 44.1kHz.

> $y = \sin(440x)$

As sample rate is number of samples taken per second, number of samples can be found out with the help of duration and sample rate:

> number of samples = (sample rate) \* (duration)

`angle` is the angular increment per sample.

`sample` is snapshot of the wave characteristics at that moment, in this case it is the amplitude i.e. value of the function at that point.

The sample (which is a floating point number) is converted to its corresponding little-endian byte representation and then written to `wave.bin`. I'm converting it to little-endian as my CPU (Intel i5) uses little-endian. Check which byte representation your machine's CPU follows via the following command:

```bash
lscpu | grep "Byte Order"
```

To play the audio, we can use `ffplay`

```bash
ffplay -f f32le -ar 44100 -showmode 1 wave.bin
```

1. `-f` specifies the file format. `f32le` indicates that the audio is encoded in 32-bit litte-endian byte format.

2. `-ar` specifies the audio sample rate, which is 44.1 kHz in this case.

3. `-showmode 1` opens a GUI showing the sine wave re-created from the samples.

On running the above `ffplay` command, you should hear a sound something similar to - [rec.mp4](https://files.catbox.moe/b8n1k9.mp4)

## Adding exponential decay

Right now, the audio abruptly ends. Let's fix that by adding exponential decay. Exponential decay keeps on gradually decreasing the amplitude, which leads to a neat _fade away_ sorta effect.

```go
startAmplitude := 1.0
endAmplitude := 1.0e-4
decayFactor := math.Pow(endAmplitude/startAmplitude, 1.0/float64(ns))
// ...

for i := 0; i < ns; i++ {
  sample := math.Sin(angle * freq * float64(i))
  sample *= startAmplitude
  startAmplitude *= decayFactor

  var buf [4]byte

  binary.LittleEndian.PutUint32(buf[:], math.Float32bits(float32(sample)))

  if _, err := f.Write(buf[:]); err != nil {
    panic(err.Error())
  }
}
```

On running the script, you should notice that the audio fades off that the end - [rec.mp4](https://files.catbox.moe/lcls0l.mp4)

## Understanding wave file format

We have generated the byte code that can produce some sound, let's save it into a `wave` file, so it can be played by the media players rather than using `ffplay`.

Waveform audio file format or wave in short stores audio data as samples, along with some metadata such as number of audio channels (mono, stereo, etc.). A wave file is usually encoded using [pulse code modulation](https://www.sciencedirect.com/topics/engineering/pulse-code-modulation) (although, it isn't required to fully understand pulse code modulation to implement this blog post by yourself).

A wave file follows a strict format and is majorly split into three _blocks_ of data.

1. header
2. fmt - holds the related metadata
3. raw data

---

The structure of the header is as follows: (on the left side, byte offsets are mentioned and on the right side, the corresponding data's label)

1. 0 - 4 bytes - chunk id (must be equal to `RIFF`, written in little-endian. if it was written in big-endian, then it would be have been `RIFX`)

2. 4 - 8 bytes - chunk size

3. 8 - 12 bytes - format (must be equal to `WAVE`)

---

The structure of the "fmt" block is as follows:

1. 12 - 16 - sub chunk 1 id (must be equal to `fmt `)

2. 16 - 20 - sub chunk 1 size

3. 20 - 22 - audio format (equal to `1`, if it is encoded via pulse code modulation)

4. 22 - 24 - number of audio channels (1 for mono, 2 for stereo)

5. 24 - 28 - sample rate (i.e. sample frequency)

6. 28 - 32 - byte rate (byte rate = (sample rate) _ (number of audio channels) _ (bits per sample) / 8)

7. 32 - 34 - block align (block align = (number of audio channels) \* (bits per sample) / 8)

8. 34 - 36 - bits per sample (i.e. bit depth)

---

The structure of the data block is as follows:

1. 36 - 40 - sub chunk 2 id (must be equal to `data`)

2. 40 - 44 - sub chunk 2 size

3. 44 - ... - raw samples

---

Here is a better pictorial representation of the structure - [https://ccrma.stanford.edu/courses/422/projects/WaveFormat/](https://web.archive.org/web/20141213140451/https://ccrma.stanford.edu/courses/422/projects/WaveFormat/). If you wanna go a bit deeper, then you can play around by opening a `wave` file in hex editors like [ImHex](https://imhex.werwolv.net/).

## Working with Wave Files

Let's first code out some structs adhering to the above mentioned format.

```go
package types

type Sample float64

type WaveHeader struct {
  ChunkId   []byte
  ChunkSize int
}

type WaveFmt struct {
  SubChunk1Id   []byte
  SubChunk1Size int
  AudioFormat   int
  NumOfChannels int
  SampleRate    int
  ByteRate      int
  BlockAlign    int
  BitsPerSample int
}
```

Before coding out an implementation for `WaveWriter`, let's make a few utility functions which would convert ints/floats to their little-endian byte representations.

```go
func IntToBits(i int, size int) []byte {
  switch size {
  case 16:
    return Int16ToBits(i)
  case 32:
    return Int32ToBits(i)
  default:
    panic("invalid size. only 16 and 32 bits are accepted")
  }
}

func Int16ToBits(i int) []byte {
  b := make([]byte, 2)
  binary.LittleEndian.PutUint16(b, uint16(i))
  return b
}

func Int32ToBits(i int) []byte {
  b := make([]byte, 4)
  binary.LittleEndian.PutUint32(b, uint32(i))
  return b
}

func FloatToBits(f float64, size int) []byte {
  bits := math.Float64bits(f)
  b := make([]byte, 8)
  binary.LittleEndian.PutUint64(b, bits)

  switch size {
  case 2:
    return b[:2]
  case 4:
    return b[:4]
  }

  return b
}
```

And a few more utility functions which would convert `WaveHeader` and `WaveFmt` into their equivalent litte-endian byte representation.

```go
func WaveFmtToBits(wfmt types.WaveFmt) []byte {
  var b []byte

  b = append(b, wfmt.SubChunk1Id...)
  b = append(b, Int32ToBits(wfmt.SubChunk1Size)...)
  b = append(b, Int16ToBits(wfmt.AudioFormat)...)
  b = append(b, Int16ToBits(wfmt.NumOfChannels)...)
  b = append(b, Int32ToBits(wfmt.SampleRate)...)
  b = append(b, Int32ToBits(wfmt.ByteRate)...)
  b = append(b, Int16ToBits(wfmt.BlockAlign)...)
  b = append(b, Int16ToBits(wfmt.BitsPerSample)...)

  return b
}

func SamplesToBits(samples []types.Sample, wfmt types.WaveFmt) ([]byte, error) {
  var b []byte

  for _, s := range samples {
    var multiplier int

    switch wfmt.BitsPerSample {
    case 8:
      multiplier = math.MaxInt8
    case 16:
      multiplier = math.MaxInt16
    case 32:
      multiplier = math.MaxInt32
    case 64:
      multiplier = math.MaxInt64
    default:
      return nil, fmt.Errorf("invalid size - %d, must be 8, 16, 32 or 64-bits only", wfmt.BitsPerSample)
    }

    bits := IntToBits(int(float64(s)*float64(multiplier)), wfmt.BitsPerSample)
    b = append(b, bits...)
  }

  return b, nil
}

func CreateHeaderBits(samples []types.Sample, wfmt types.WaveFmt) []byte {
  var b []byte

  chunkSizeInBits := Int32ToBits(36 + (len(samples)*wfmt.NumOfChannels*wfmt.BitsPerSample)/8)

  b = append(b, []byte(constants.WaveChunkId)...)
  b = append(b, chunkSizeInBits...)
  b = append(b, []byte(constants.WaveFileFormat)...)

  return b
}
```

Now as we have all the utility functions set up, let's write a simple struct `WaveWriter` which implements a method `WriteWaveFile` which would save the sample data into a wave file.

```go
type WaveWriter struct{}

func NewWaveWriter() WaveWriter {
  return WaveWriter{}
}

func (w WaveWriter) WriteWaveFile(file string, samples []types.Sample, metadata types.WaveFmt) error {
  f, err := os.Create(file)
  if err != nil {
    return err
  }
  defer f.Close()

  var data []byte

  headerBits := utils.CreateHeaderBits(samples, metadata)
  data = append(data, headerBits...)

  wfmtInBits := utils.WaveFmtToBits(metadata)
  data = append(data, wfmtInBits...)

  data = append(data, []byte(constants.WaveSubChunk2Id)...)
  data = append(data, utils.Int32ToBits(len(samples)*metadata.NumOfChannels*metadata.BitsPerSample/8)...)

  samplesBits, err := utils.SamplesToBits(samples, metadata)
  if err != nil {
    return err
  }
  data = append(data, samplesBits...)

  if _, err := f.Write(data); err != nil {
    return err
  }

  return nil
}
```

The header is created using the above utility function (`CreateHeaderBits`) and also the fmt block and samples are converted to their equivalent byte representations using the above utility functions (`WaveFmtToBits` and `SamplesToBits`).

Let's use the `WriteWaveFile` method in our script to save the samples into a `wave` file.

```go
var samples []types.Sample

for i := 0; i < ns; i++ {
  sample := types.Sample(math.Sin(angle*freq*float64(i)) * startAmplitude)
  startAmplitude *= decayFactor

  samples = append(samples, sample)
}

waveWriter := helpers.NewWaveWriter()
if err := waveWriter.WriteWaveFile("test.wav", samples, wavefmt); err != nil {
  panic(err.Error())
}
```

And on running the script, a new file named `test.wav` must be created, which must sound similar to - [test.wav](https://files.catbox.moe/bq0vik.wav)

## Reading wave files

We've successfully implemented the first part of the blog post which is to generate a sine wave of a constant frequency and save it to a wave file. The next part is to amplify a given wave file. To amplify a given input wave file, we have to first parse through it. To do so, we have to implement `WaveReader`.

Before actually implementing `WaveReader`, we have to add a few utility functions which convert bits to ints/floats.

```go
func BitsToInt(b []byte, size int) int {
  switch size {
  case 16:
    return Bits16ToInt(b)
  case 32:
    return Bits32ToInt(b)
  default:
    panic("invalid size. only 16 and 32 bits are accepted")
  }
}

func Bits16ToInt(b []byte) int {
  if len(b) != 2 {
    panic(fmt.Errorf("invalid size. expected 2, got %d", len(b)))
  }

  var payload int16
  buf := bytes.NewReader(b)
  if err := binary.Read(buf, binary.LittleEndian, &payload); err != nil {
    panic(err.Error())
  }

  return int(payload)
}

func Bits32ToInt(b []byte) int {
  if len(b) != 4 {
    panic(fmt.Errorf("invalid size. expected 4, got %d", len(b)))
  }

  var payload int32
  buf := bytes.NewReader(b)
  if err := binary.Read(buf, binary.LittleEndian, &payload); err != nil {
    panic(err.Error())
  }

  return int(payload)
}

func BitsToFloat(b []byte) float64 {
  switch len(b) {
  case 4:
    bits32 := binary.LittleEndian.Uint32(b)
    return float64(math.Float32frombits(bits32))
  case 8:
    bits64 := binary.LittleEndian.Uint64(b)
    return math.Float64frombits(bits64)
  default:
    panic(fmt.Errorf("invalid size: %d, must be 32 or 64 bits", len(b)*8))
  }
}
```

The `WaveReader` struct implements a few methods which parse through the input `wave` file and return parsed header (`WaveHeader`), fmt block (`WaveFmt`) and samples (`[]Samples`). While parsing, the samples are scaled down as during writing, the samples were multiplied with max value of that corresponding bit length so it must be divided while parsing to maintain consistency.

```go
type WaveReader struct{}

func NewWaveReader() WaveReader {
  return WaveReader{}
}

func (r WaveReader) ParseFile(file string) (types.Wave, error) {
  f, err := os.Open(file)
  if err != nil {
    return types.Wave{}, err
  }
  defer f.Close()

  data, err := io.ReadAll(f)
  if err != nil {
    return types.Wave{}, err
  }

  header, err := r.parseHeader(data)
  if err != nil {
    return types.Wave{}, err
  }

  wavefmt, err := r.parseMetadata(data)
  if err != nil {
    return types.Wave{}, err
  }

  samples, err := r.parseData(data)
  if err != nil {
    return types.Wave{}, err
  }

  wave := types.Wave{
    WaveHeader: header,
    WaveFmt:    wavefmt,
    Samples:    samples,
  }

  return wave, nil
}

func (r WaveReader) parseHeader(data []byte) (types.WaveHeader, error) {
  header := types.WaveHeader{}

  chunkId := data[0:4]
  if string(chunkId) != constants.WaveChunkId {
    return header, errors.New("invalid file")
  }
  header.ChunkId = chunkId

  chunkSize := data[4:8]
  header.ChunkSize = utils.Bits32ToInt(chunkSize)

  format := data[8:12]
  if string(format) != constants.WaveFileFormat {
    return header, errors.New("invalid format")
  }

  return header, nil
}

func (r WaveReader) parseMetadata(data []byte) (types.WaveFmt, error) {
  metadata := types.WaveFmt{}

  subChunk1Id := data[12:16]
  if string(subChunk1Id) != constants.WaveSubChunk1Id {
    return metadata, fmt.Errorf("invalid sub chunk 1 id - %s", string(subChunk1Id))
  }

  metadata.SubChunk1Id = subChunk1Id
  metadata.SubChunk1Size = utils.Bits32ToInt(data[16:20])
  metadata.AudioFormat = utils.Bits16ToInt(data[20:22])
  metadata.NumOfChannels = utils.Bits16ToInt(data[22:24])
  metadata.SampleRate = utils.Bits32ToInt(data[24:28])
  metadata.ByteRate = utils.Bits32ToInt(data[28:32])
  metadata.BlockAlign = utils.Bits16ToInt(data[32:34])
  metadata.BitsPerSample = utils.Bits16ToInt(data[34:36])

  return metadata, nil
}

func (r WaveReader) parseData(data []byte) ([]types.Sample, error) {
  metadata, err := r.parseMetadata(data)
  if err != nil {
    return nil, err
  }

  subChunk2Id := data[36:40]
  if string(subChunk2Id) != constants.WaveSubChunk2Id {
    return nil, fmt.Errorf("invalid sub chunk 2 id - %s", string(subChunk2Id))
  }

  bytesPerSampleSize := metadata.BitsPerSample / 8
  rawData := data[44:]

  samples := []types.Sample{}

  for i := 0; i < len(rawData); i += bytesPerSampleSize {
    rawSample := rawData[i : i+bytesPerSampleSize]
    unscaledSample := utils.BitsToInt(rawSample, metadata.BitsPerSample)
    scaledSample := types.Sample(float64(unscaledSample) / float64(utils.MaxValue(metadata.BitsPerSample)))
    samples = append(samples, scaledSample)
  }

  return samples, nil
}
```

If you have noticed, I have added a new type `Wave` which just contains wave header, fmt block and samples - easier to access and return the data through `Wave`.

```go
type Wave struct {
  WaveHeader
  WaveFmt
  Samples []Sample
}
```

## Amplifying a given wave file

In simple words, amplification is nothing more than scaling up/down the amplitude of the wave at a given point. So basically, messing around with each individual sample.

```go
waveReader := helpers.NewWaveReader()
waveWriter := helpers.NewWaveWriter()

wave, err := waveReader.ParseFile(input)
if err != nil {
  return err
}

var updatedSamples []types.Sample

for _, sample := range wave.Samples {
  updatedSample := types.Sample(float64(sample) * scaleFactor)
  updatedSamples = append(updatedSamples, updatedSample)
}

if err := waveWriter.WriteWaveFile(output, updatedSamples, wave.WaveFmt); err != nil {
  return err
}
```

This is the output when a 440Hz pure sine wave is scaled down by 0.2 times - [output.wav](https://files.catbox.moe/5bdo6o.wav)

## Stereopanning

In the context of audio programming, stereopanning refers to _positioning_ the sound within the space, allowing you to make it appear as if it's coming from the left speaker, right speaker, or anywhere in between by adjusting the audio signal.

Until now, we have only worked with single audio channels but in case of multiple audio channels, a single float number doesn't make up a sample but rather multiple float numbers make up a sample.

$[f_1][f_2][f_3][f_4]...[f_n]$; where $f_n$ is the nth float number

If the number of audio channels is set to be 1, then $f_1$ alone makes up for the 1st sample. Whereas, if the number of audio channels is set to be 2, then $f_1$ and $f_2$ combined make up for the 1st sample and $f_3$ and $f_4$ combined make up for the 2nd sample.

For sake of simplicity, we would perform stereopanning on a mono audio file (i.e. number of audio channels = 1) and return a stereo audio file (i.e. number of audio channels = 2).

Generally, the position of audio in the space is represented using a number within the range of [-1, 1].

1. -1 indicates that the sound is completely positioned towards the left.

2. 1 indicates that the sound is completely positioned towards the right.

3. 0 indicates that the sound is equally distributed between the two channels i.e. kept at the _center_.

Consider `p` to be the panning position in range of [-1, 1], then the multiplying factor for both the audio signals can be found by transforming the ranges.

1. For left channel, [-1, 1] is transformed to [-0.5, 0.5] by dividing by 2 and then subtracted by 0.5 to get [-1, 0]

   `p` → `p`/2 → (`p`/2) - 0.5

2. For right channel, [-1, 1] is transformed to [-0.5, 0.5] and then 0.5 is added to get [0, 1]

   `p` → `p`/2 → (`p`/2) + 0.5

In code, it can be expressed as follows:

```go
func PanPositionToChanMultipliers(p float64) (float64, float64) {
  if !(p >= -1 && p <= 1) {
    panic("pan position outside [-1, 1] range")
  }

  leftChanMultiplier := (p / 2) - 0.5
  rightChanMultiplier := (p / 2) + 0.5

  return leftChanMultiplier, rightChanMultiplier
}
```

And after figuring out the multipliers for left and right channels, it is pretty much same as amplifying -- just multiplying that factor and writing that data to the `wave` file.

```go
wave, err := waveReader.ParseFile(input)
if err != nil {
  return err
}

leftChanMultiplier, rightChanMultiplier := utils.PanPositionToChanMultipliers(panningPosition)

var updatedSamples []types.Sample

for _, sample := range wave.Samples {
  updatedSamples = append(updatedSamples, types.Sample(sample.ToFloat()*leftChanMultiplier))
  updatedSamples = append(updatedSamples, types.Sample(sample.ToFloat()*rightChanMultiplier))
}

wave.WaveFmt.NumOfChannels = 2

if err := waveWriter.WriteWaveFile(output, updatedSamples, wave.WaveFmt); err != nil {
  return err
}
```

I ran the script on [rec.wav](https://files.catbox.moe/6f5yfi.wav) and here is the output when the panning position is equal to -1 - [left.wav](https://files.catbox.moe/3zo6jl.wav) and here is the output when it is equal to 1 - [right.wav](https://files.catbox.moe/ki3ng4.wav)

You can clearly notice that when you play `left.wav`, the audio just comes from the left speaker and when you play `right.wav`, it just comes from the right speaker.

...and well that is pretty much it for this blog post. I might write a few more blog posts about this topic covering topics such as waveform tables and ADSR.

[Source code](https://github.com/0xMukesh/sound-synthesizer/)
