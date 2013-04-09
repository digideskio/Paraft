varying vec3        textureCoords;

uniform sampler3D   data;
uniform sampler1D   tf;
uniform sampler1D   tfMapping;
uniform sampler2D   preInt;
uniform sampler2D   preIntFront;
uniform sampler2D   preIntBack;
uniform vec3        viewVec;
uniform vec3        scaleDim;
uniform float       sampleSpacing;
uniform bool        enableLight;
uniform vec4        lightParam;     // r: ambient, g: diffuse, b: specular, a: shininess
uniform float       projection;     // 0: orth, 1: perspective
uniform float       mapping;        // 0: use linear mapping, other: user define mapping

#define BASESAMPLE 0.01
#define EPSILON 0.01
vec3 epsilon;

struct Light
{
    float ambientDiffuse;
    float specular;
};

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

vec4 getColor(float scalar,float spacing)
{
    vec4 tempColor = texture1D(tf, scalar);
    return tempColor;
}

vec4 getColorA(float scalar, float spacing, vec3 texPosition)
{
    vec4 tmpColor;
    if (mapping == 0.0)
        tmpColor = texture1D(tf, scalar);
    else
    {
        float realPos = texture1D(tfMapping, scalar).r;
        tmpColor = texture1D(tf, realPos);
    }
    float factor = spacing / BASESAMPLE;
    float adjAlpha = 1.0 - pow(1.0 - tmpColor.a, factor);
    tmpColor.a = adjAlpha;
    tmpColor.rgb = tmpColor.rgb * tmpColor.a;
    return tmpColor;
}

vec4 getPreIntColor(float fscalar, float bscalar)
{
    vec4 color = texture2D(preInt, vec2(fscalar, bscalar));
    //color.rgb = color.rgb * color.a;
    return color;
}

void main()
{
    vec4 color = vec4(0.0);
    //vec3 samplePos = vec3(0.0);
    vec3 rayStart = textureCoords;
    vec3 tex = textureCoords;
    vec3 view = viewVec;

    vec3 scale = scaleDim;
    rayStart /= scale;
    if (projection == 1.0)  // perspective
        view = tex - view;  // for perspective, viewVec is actually a "camera position"
    view = normalize(view);
    vec3 viewObj = view;    // viewObj is the view vector in object coordinate space
    view /= scale;          // view is the view vector in texture coordinate space

    epsilon = vec3(1.0) / scale * EPSILON;
    
    float sampleLen = 0.0;
    //float normalizedStep = length(scaleDim);
    //float delta = sampleSpacing * normalizedStep;
    float delta = sampleSpacing;
    vec4 sampleColor = vec4(0.0);
    //float maxScalar = -100.0;

    float fscalar, bscalar;
    vec3 fpos, bpos;

    //samplePos = rayStart + sampleLen * view;
    fpos = rayStart + sampleLen * view;
    fscalar = texture3D(data, fpos).s;
    sampleLen += delta;

    int count = 0;

    while (true)
    {
        //samplePos = rayStart + sampleLen * view;
        bpos = rayStart + sampleLen * view;
        bscalar = texture3D(data, bpos).s;

        sampleLen += delta;
        sampleColor = getPreIntColor(fscalar, bscalar);
        vec4 fcolor = texture2D(preIntFront, vec2(fscalar, bscalar));
        vec4 bcolor = texture2D(preIntBack, vec2(fscalar, bscalar));

        if (sampleColor.a > 0.001)
        {
            if (enableLight)
            {
                /*vec3 normal = getNormal(fpos);
                vec3 lightDir = normalize(vec3(-viewObj));
                Light oneLight = getLight(normal, lightDir, viewObj);
                float ambientDiffuse = oneLight.ambientDiffuse;
                float specular = oneLight.specular;
                sampleColor.rgb = sampleColor.rgb * ambientDiffuse + sampleColor.a * specular;*/

                vec3 normal = getNormal(fpos);
                vec3 lightDir = normalize(vec3(-viewObj));
                Light oneLight = getLight(normal, lightDir, viewObj);
                float ambientDiffuse = oneLight.ambientDiffuse;
                float specular = oneLight.specular;
                //vec4 fcolor = texture2D(preIntFront, vec2(fscalar, bscalar));
                fcolor.rgb = fcolor.rgb * ambientDiffuse + fcolor.a * specular;

                normal = getNormal(bpos);
                lightDir = normalize(vec3(-viewObj));
                oneLight = getLight(normal, lightDir, viewObj);
                ambientDiffuse = oneLight.ambientDiffuse;
                specular = oneLight.specular;
                //vec4 bcolor = texture2D(preIntBack, vec2(fscalar, bscalar));
                bcolor.rgb = bcolor.rgb * ambientDiffuse + bcolor.a * specular;

                sampleColor.rgb = fcolor.rgb + bcolor.rgb;
            }

            color.rgb += (1.0 - color.a) * sampleColor.rgb;
            color.a   += (1.0 - color.a) * sampleColor.a;
        }

        /*if (count++ >= 1000)
        {
            color.a = 1.0;
            break;
        }*/

        if (color.a > 0.999)
            break;
        //if (samplePos.x >  1.01 || samplePos.y >  1.01 || samplePos.z >  1.01 ||
        //    samplePos.x < -0.01 || samplePos.y < -0.01 || samplePos.z < -0.01)
        if (bpos.x >  1.0001 || bpos.y >  1.0001 || bpos.z >  1.0001 ||
            bpos.x < -0.0001 || bpos.y < -0.0001 || bpos.z < -0.0001)
            break;

        fscalar = bscalar;
        fpos = bpos;
    }
    
    /*while (true)
    {
        samplePos = rayStart + sampleLen * view;
        float scalar = texture3D(data, samplePos).s;

        sampleLen += delta;
        sampleColor = getColorA(scalar, sampleSpacing, samplePos);

        if (sampleColor.a > 0.001)
        {
            if (enableLight)
            {
                vec3 normal = getNormal(samplePos);
                vec3 lightDir = normalize(vec3(-viewObj));
                Light oneLight = getLight(normal, lightDir, viewObj);
                float ambientDiffuse = oneLight.ambientDiffuse;
                float specular = oneLight.specular;
                sampleColor.rgb = sampleColor.rgb * ambientDiffuse + sampleColor.a * specular;
            }

            color.rgb += (1.0 - color.a) * sampleColor.rgb;
            color.a   += (1.0 - color.a) * sampleColor.a;
        }


        //if(maxScalar < scalar){
            //maxScalar = scalar;
        //}
        

        if (color.a > 0.999)
            break;
        if (samplePos.x >  1.01 || samplePos.y >  1.01 || samplePos.z >  1.01 ||
            samplePos.x < -0.01 || samplePos.y < -0.01 || samplePos.z < -0.01)
            break;
    }*/

    gl_FragColor = color;
}
