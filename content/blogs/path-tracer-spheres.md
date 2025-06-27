---
title: Making a path tracer - Working with spheres
draft: false
date: 2024-12-25
---

In the [previous](https://0xmukesh.bearblog.dev/writing-a-mini-path-tracer-making-a-gradient/) blog post of this series, we've setup a simple virtual camera and have rendered a blue-white gradient. In this blog, we will render a simple sphere and learn about antialiasing.

## Some math related to spheres

We have to first figure out whether a certain ray _hits_ a sphere or not.

The equation for a sphere of radius `r` and centered at origin can be mathematically represented as follows:

> $x^2 + y^2 + z^2 = r^2$

And position of any arbitary point (a, b, c) with respect to the sphere can be judged by comparing the value of $a^2 + b^2 + c^2$ with $r^2$

1. If $a^2 + b^2 + c^2 = r^2$, then it lies on the surface of the sphere
2. If $a^2 + b^2 + c^2 > r^2$, then it lies outside the sphere
3. If $a^2 + b^2 + c^2 < r^2$, then it lies within the sphere

The equation for a sphere of radius `r` and centered at a point $(C_x, C_y, C_z)$ is as follows:

> $(C_x - x)^2 + (C_y - y)^2 + (C_z - z)^2 = r^2$

The above equation can be also represented in the form of vectors as follows:

> $C = (C_x, C_y, C_z)$

> $P = (x, y, z)$

> $(C - P) \cdot (C - P) = r^2$

So to check if a ray _hits_ the sphere, we've to find the value of `t` which satisfies the following equation

> $P(t) = Q + td$, where $Q$ is the origin of the ray and $d$ is the direction of the ray

> $(C - P(t)) \cdot (C - P(t)) = r^2$

After doing some math magic, you realize that it is a quadratic equation in `t`

> $(d \cdot d)t^2 - (2d \cdot (C - Q))t + (C - Q) \cdot (C - Q) - r^2 = 0$

And from algebra, a quadratic equation has real solutions only if its discriminant is greater than or equal to 0

> $D = b^2 - 4ac$

> $D = (2d \cdot (C - Q))^2 - 4(d \cdot d)((C - Q) \cdot (C - Q) - r^2)$

To render a red sphere, we've to write some logic within `RayColor` function which would return red color if the discriminant is greater than or equal to 0 else it would return the color related to the background. So let's implement it in code.

```go
func (r Ray) HitSphere(center Vector, radius float64) bool {
  cq := center.SubtractVector(r.Origin)
  a := r.Direction.DotProduct(r.Direction)
  b := 2 * r.Direction.DotProduct(cq)
  c := cq.DotProduct(cq) - radius*radius
  d := b*b - 4*a*c

  return d >= 0
}

func RayColor(r Ray) Vector {
  if r.HitSphere(NewVector(0, 0, -1), 0.5) {
    return NewVector(1, 0, 0)
  }

  unitVec := r.Direction.UnitVector()
  t := 0.5 * (unitVec.Y + 1)
  color := NewVector(1, 1, 1).MultiplyScalar(1.0 - t).AddVector(NewVector(0.5, 0.7, 1.0).MultiplyScalar(t))
  return color
}
```

On running the script, you must see a red sphere in center of the image

![](https://i.imgur.com/83bOrp9.png)

And this is your first raytraced image. Congo!

## Surface Normals

Normals are very much crucial for shading in ray tracing because they represent the surface orientation at each point, determining how light interacts with the object and also calculate the brightness of a surface

Let's make a simple algorithm which shades the sphere based on the surface normals. The first part of shading a sphere based on surface normal is to calculate direction of the surface normal. Calculating direction of a normal is pretty much straightforward.

> $N = P - C$, where $P$ is the point of intersection of the ray with the sphere

`N` always points in the direction outside of the sphere

```go
func (r Ray) HitSphere(center Vector, radius float64) (bool, float64) {
  cq := center.SubtractVector(r.Origin)
  a := r.Direction.DotProduct(r.Direction)
  b := 2 * r.Direction.DotProduct(cq)
  c := cq.DotProduct(cq) - radius*radius
  d := b*b - 4*a*c

  if d >= 0 {
    p := (-b - math.Sqrt(d)) / (2 * a)
    return true, p
  }

  return false, 0
}

func RayColor(r Ray) Vector {
  center := NewVector(0, 0, -1)
  radius := 0.5

  found, p := r.HitSphere(center, radius)

  if found {
    normal := r.At(p).SubtractVector(center).UnitVector()
    return NewVector(normal.X+1, normal.Y+1, normal.Z+1).MultiplyScalar(0.5)
  }

  unitVec := r.Direction.UnitVector()
  t := 0.5 * (unitVec.Y + 1)
  color := NewVector(1, 1, 1).MultiplyScalar(1.0 - t).AddVector(NewVector(0.5, 0.7, 1.0).MultiplyScalar(t))
  return color
}
```

I've updated the `HitSphere` function a bit to also return the solution of that quadratic equation, which is later used by `Ray.At` method to calculate the surface normal.

```go
if d >= 0 {
  p := (-b - math.Sqrt(d)) / (2 * a)
  return true, p
}
```

If you have noticed, the `HitSphere` function only takes one of the roots into consideration. This is just to simplify things, as we have only spheres in front of the camera. We will just consider the closest hit point value i.e. the smallest value of `p`.

```go
if found {
  normal := r.At(p).SubtractVector(center).UnitVector()
  return NewVector(normal.X+1, normal.Y+1, normal.Z+1).MultiplyScalar(0.5)
}
```

The shading process is pretty much simple. An unit vector is found along the direction of the normal and each component of the unit vector is mapped to a range of 0.0 to 1.0.

On running the script, the rendered image must be similar to

![](https://i.imgur.com/MUrvejw.png)

## Antialiasing

In computer graphics, aliasing is the effect that causes signal to become indistinguishable from one another.

If you look closely at the edges of the sphere, you notice that they look very jaggy. This is because our path tracer still only colors the pixels based on whether the ray hit our object. There is no in-between. Due to which, the rendered sphere does not blend into the background smoothly as it would if it were being viewed in the real world. This can be fixed by introducing sampling.

In our case, sampling is just taking a number of different samples and averaging them together to get the final result. This would produce a much more realistic image and reduce the _jagginess_

```go
for j := imageHeight - 1; j >= 0; j-- {
  fmt.Printf("scanlines remaining: %d\n", j)
  for i := 0; i < imageWidth; i++ {
    rgb := Vector{}

    for s := 0; s < samplesPerPixel; s++ {
      u := (float64(i) + rand.Float64()) / float64(imageWidth-1)
      v := (float64(j) + rand.Float64()) / float64(imageHeight-1)

      vec := lowerLeftCorner.AddVector(horizontal.MultiplyScalar(u)).AddVector(vertical.MultiplyScalar(v)).SubtractVector(origin)
      ray := NewRay(origin, vec)

      color := RayColor(ray)
      rgb = rgb.AddVector(color)
    }

    rgb = rgb.DivideScalar(float64(samplesPerPixel))

    ir := int(255.99 * rgb.X)
    ig := int(255.99 * rgb.Y)
    ib := int(255.99 * rgb.Z)

    _, err := fmt.Fprintf(f, "%d %d %d\n", ir, ig, ib)
    if err != nil {
      panic(err.Error())
    }
  }
}
```

Instead of casting just one ray per pixel, multiple rays are casted and then their results are averaged out.

For each sample, a random offset (in the range of [0, 1)) is added to the pixel coordinates. So each time, the ray is shot to a slightly different point within the pixel's area.

This random sampling softens the jagged edges and creates much more smoother gradients in areas with changing colors or lighting.

`samplesPerPixel` is the number of rays which are processed to render a single pixel. The processed value of each ray is called sample and these samples are averaged out to get the final color for the pixel.

On running the script with `samplesPerPixel` equal to 100, the rendered image should look something similar to

![](https://i.imgur.com/hHYCLmv.png)

Notice how the edges are much more smoother now, but with sampling, the processing time increases a lot as multiple rays are to be processed to render a single pixel. It took ~1.25 seconds to render a 256x256 pixel image (100 rays were processed and averaged out to render a single pixel).
