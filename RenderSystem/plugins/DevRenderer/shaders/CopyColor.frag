uniform sampler2D colorBuffer;
uniform vec2      invImageDim;      // (1.0 / width, 1.0 / height)

void main()
{
    gl_FragColor = texture2D(colorBuffer, gl_FragCoord.xy * invImageDim);
}
