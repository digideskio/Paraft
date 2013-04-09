//varying vec3        textureCoords;
varying vec3        hitPoint;       // hit point in world (= object) space

uniform sampler3D   data;           // scalar data
uniform sampler1D   tf;             // transfer function
uniform sampler2D   colorBuffer;    // color accumulated in previous subblocks (front-to-back)
uniform vec2        imageScale;     // (1 / width, 1 / height)
////uniform sampler1D   tfMapping;
uniform vec3        viewVec;        // ray origin (perspective)
//uniform vec3        scaledDim;
uniform vec3        boxLo;          // subblock box
uniform vec3        boxHi;
//uniform vec3        scaledPaddedLo;
//uniform vec3        scaledPaddedDim;
uniform vec3        offset;         // p_tex = p_obj * scale + offset
uniform vec3        scale;          // v_tex = v_obj * scale
//uniform float       sampleSpacing;
uniform float       sampleInterval;
uniform bool        lightEnabled;
uniform vec4        lightParam;     // r: ambient, g: diffuse, b: specular, a: shininess
uniform float       projection;     // 0: orth, 1: perspective
////uniform float       mapping;        // 0: use linear mapping, other: user define mapping

#define BASESAMPLE 0.01
#define EPSILON 0.01

vec3 epsilon;

struct Light
{
    float ambientDiffuse;
    float specular;
};

void rayBoxIntersect(in vec3 origin, in vec3 dir, out float tnear, out float tfar)
{
    if (dir.x == 0.0) dir.x = 1.0e-10;      // avoid boundary case
    if (dir.y == 0.0) dir.y = 1.0e-10;
    if (dir.z == 0.0) dir.z = 1.0e-10;
    vec3 v1 = (boxLo - origin) / dir;
    vec3 v2 = (boxHi - origin) / dir;
    vec3 mins = min(v1, v2);
    vec3 maxes = max(v1, v2);
    tnear = max(mins.x, max(mins.y, mins.z));
    tfar = min(maxes.x, min(maxes.y, maxes.z));
}

Light getLight(vec3 normal, vec3 lightDir, vec3 rayDir)
{
    Light light;
    float ambient = lightParam.r;
    float diffuse = lightParam.g * max(dot(lightDir, normal), dot(lightDir, -normal));
    vec3 H = normalize(-rayDir + lightDir);
    float dotHV = max(dot(H, normal), dot(H, -normal));
    float specular = 0.0;
    if (dotHV > 0.0)
        specular = lightParam.b * pow(dotHV, lightParam.a);
    light.ambientDiffuse = ambient + diffuse;
    light.specular = specular;
    return light;
}

vec3 getNormal(vec3 texPosition)
{
    vec3 gradient;
    gradient.x = texture3D(data, texPosition.xyz + vec3(epsilon.x, 0, 0)).r -
                 texture3D(data, texPosition.xyz + vec3(-epsilon.x, 0, 0)).r;
    gradient.y = texture3D(data, texPosition.xyz + vec3(0, epsilon.y, 0)).r -
                 texture3D(data, texPosition.xyz + vec3(0, -epsilon.y, 0)).r;
    gradient.z = texture3D(data, texPosition.xyz + vec3(0, 0, epsilon.z)).r -
                 texture3D(data, texPosition.xyz + vec3(0, 0, -epsilon.z)).r;
    if (length(gradient) > 0.0)
        gradient = normalize(-gradient);
    return gradient;
}

/*vec4 getColor(float scalar, float spacing)
{
    vec4 tempColor = texture1D(tf, scalar);
    return tempColor;
}*/

vec4 getColorA(float scalar, float interval, vec3 texPosition)
{
    vec4 tmpColor = texture1D(tf, scalar);
    float factor = interval / BASESAMPLE;
    float adjAlpha = 1.0 - pow(1.0 - tmpColor.a, factor);
    tmpColor.a = adjAlpha;
    tmpColor.rgb = tmpColor.rgb * tmpColor.a;
    return tmpColor;
}

void main()
{
    //epsilon = vec3(1.0) / scaledPaddedDim * EPSILON;
    epsilon = vec3(1.0, 1.0, 1.0) * scale * EPSILON;

    vec4 color = texture2D(colorBuffer, gl_FragCoord.xy * imageScale);

    if (color.a > 0.999)
    {
        gl_FragColor = color;
        return;
    }

    //vec3 samplePos = vec3(0.0);
    //vec3 rayStart = hitPoint;
    //vec3 tex = hitPoint;
    //vec3 view = viewVec;

    //vec3 scale = scaledDim;

    // world -> texture space
    //rayStart = (rayStart - scaledPaddedLo) / scaledPaddedDim;

    //if (projection == 1.0)  // perspective
    //    view = hitPoint - view;     // for perspective, viewVec is actually a "camera position"
    //view = normalize(view);
    //vec3 viewObj = view;    // viewObj is the view vector in object coordinate space
    //view /= scaledPaddedDim;    // view is the view vector in texture coordinate space

    vec3 rayOrigin = viewVec;                       // ray origin in object space
    vec3 rayDir = normalize(hitPoint - viewVec);    // ray dir in object space

    //float sampleLen = 0.0;
    //float normalizedStep = length(scaledDim);
    //float delta = sampleSpacing * normalizedStep;
    //float delta = sampleInterval;
    //vec4 sampleColor = vec4(0.0);

    float tnear, tfar;                              // intersection point of the ray and the subblock box
    rayBoxIntersect(rayOrigin, rayDir, tnear, tfar);
    float tHit = distance(hitPoint, rayOrigin);     // hit t of the whole box
    tnear = max(tnear, tHit);
    if (tnear > tfar)
    {
        //gl_FragColor = vec4(0.0, 0.5, 0.0, 1.0);
        gl_FragColor = color;
        return;
    }

    ////
    //if (boxLo.x < 0.25)
    //{
    //    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
    //    return;
    //}

    ////
    tnear = ceil((tnear - 1.0e-6 - tHit) / sampleInterval) * sampleInterval + tHit;

    //vec3 origin = (rayOrigin - scaledPaddedLo) / scaledPaddedDim;
    //vec3 dir = rayDir / scaledPaddedDim;
    vec3 origin = rayOrigin * scale + offset;       // ray origin in texture space
    vec3 dir = rayDir * scale;                      // ray dir in texture space

    while (tnear < tfar)
    {
        //samplePos = (viewVec + tnear * viewObj - scaledPaddedLo) / scaledPaddedDim;
        vec3 samplePos = origin + tnear * dir;

        //tnear += delta;
        tnear += sampleInterval;

        //samplePos = rayStart + sampleLen * view;
        float scalar = texture3D(data, samplePos).r;


        //sampleColor = getColorA(scalar, sampleSpacing, samplePos);
        vec4 sampleColor = getColorA(scalar, sampleInterval, samplePos);

        if (sampleColor.a > 0.001)
        {
            if (lightEnabled)
            {
                vec3 normal = getNormal(samplePos);
                vec3 lightDir = normalize(vec3(-rayDir));
                Light oneLight = getLight(normal, lightDir, rayDir);
                float ambientDiffuse = oneLight.ambientDiffuse;
                float specular = oneLight.specular;
                sampleColor.rgb = sampleColor.rgb * ambientDiffuse + sampleColor.a * specular;
            }

            color.rgb += (1.0 - color.a) * sampleColor.rgb;
            color.a   += (1.0 - color.a) * sampleColor.a;
        }

        if (color.a > 0.999)
            break;
        //if (samplePos.x >  1.01 || samplePos.y >  1.01 || samplePos.z >  1.01 ||
        //    samplePos.x < -0.01 || samplePos.y < -0.01 || samplePos.z < -0.01)
        //    break;
    }

    gl_FragColor = color;
}
