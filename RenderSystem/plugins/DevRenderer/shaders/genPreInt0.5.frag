uniform sampler1D   tf;

uniform float       invRes;     // inversed TF resolution (ex: 1.0 / 1024)
uniform float       adjFactor;  // = stepSize / baseStepSize (ex: 0.001 / 0.01)

void main()
{
    float beginPos = gl_FragCoord.x * invRes;
    float delta = 1.0 * invRes;
    float dist = abs(gl_FragCoord.y - gl_FragCoord.x) * invRes;
    float dir = sign(gl_FragCoord.y - gl_FragCoord.x);
    float adjExp = adjFactor / abs(gl_FragCoord.y - gl_FragCoord.x);

    vec4 color = vec4(0.0);
    vec4 front = vec4(0.0);
    vec4 back = vec4(0.0);

    if (dist < 0.5 * delta)     // dist == 0
    {
        color = texture1D(tf, beginPos);
        color.a = 1.0 - pow(1.0 - color.a, adjFactor);
        color.rgb *= color.a;
        front = color;
        back = vec4(0.0);
    }
    else
    {
        for (float t = 0.5 * delta; t < dist; t += delta)     // 0.5..dist-0.5
        {
            vec4 c = texture1D(tf, beginPos + t * dir);
            c.a = 1.0 - pow(1.0 - c.a, adjExp);
            c.a = (1.0 - color.a) * c.a;
            c.rgb *= c.a;
            color += c;
            float b = t / dist;
            front += (1.0 - b) * c;
            back += b * c;
        }
    }

    gl_FragData[0] = color;
    gl_FragData[1] = front;
    gl_FragData[2] = back;
}
