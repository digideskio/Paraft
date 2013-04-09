varying vec3        textureCoords;

uniform sampler3D   data;
uniform sampler1D   tf;
uniform sampler1D   tfMapping;
uniform vec3        viewVec;
uniform vec3        scaleDim;
uniform float       sampleSpacing;
uniform bool        enableLight;
uniform vec4        lightParam;     // r: ambient, g: diffuse, b: specular, a: shininess
uniform float       projection;     // 0: orth, 1: perspective
uniform float       mapping;        // 0: use linear mapping, other: user define mapping

uniform int         sliceNum;
uniform vec3        sliceVec[10];
uniform vec3        slicePnt[10];

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

void main()
{
    vec4 color = vec4(0.0);
    vec3 samplePos = vec3(0.0);
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
    float normalizedStep = length(scaleDim);
    float delta = sampleSpacing * normalizedStep;
    vec4 sampleColor = vec4(0.0);
    //float maxScalar = -100.0;
    
    while (true)
    {
        samplePos = rayStart + sampleLen * view;
        float scalar = texture3D(data, samplePos).s;

        sampleLen += delta;
        sampleColor = getColorA(scalar, sampleSpacing, samplePos);

        bool outside = false;
        for (int i = 0; i < sliceNum; i++)
        {
            if (dot(samplePos * scale - slicePnt[i], sliceVec[i]) > 0.00001)
            {
                outside = true;
                break;
            }
        }
        if (outside) break;

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
    }
    //sampleColor = getColor(maxScalar,sampleSpacing);
    //sampleColor.a = 0.5;

    gl_FragColor = color;
}
