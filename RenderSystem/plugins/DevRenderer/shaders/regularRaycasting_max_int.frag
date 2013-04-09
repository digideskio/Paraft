varying vec3        textureCoords;
uniform sampler1D   tf;
uniform sampler1D   tfMapping;
uniform sampler3D   data;
uniform vec3        viewVec;
uniform vec3        scaleDim;
uniform float       sampleSpacing;
uniform float       enableLight;
uniform vec4        lightParam;     // r: ambient, g: diffuse, b: specular, a: shininess
uniform float       projection;     // 0: orth, 1: perspective
uniform float       mapping;        // 0: use linear mapping, other: user define mapping

vec4 getColor(float scalar,float spacing)
{
    vec4 tempColor = texture1D(tf, scalar);
    return tempColor;
}

void main()
{
    vec3 samplePos = vec3(0.0);
    vec3 rayStart = textureCoords;
    vec3 tex = textureCoords;
    vec3 scale = scaleDim;
    float normalizedStep = length(scaleDim);
    vec3 view = viewVec;
    rayStart /= scale;
    if(projection == 1.0) //perspective
        view = tex - view; //for perspective, viewvec is actually a "camera position"
    view = normalize(view);
    vec3 viewObj = view; // viewObj is the view vector in object coordinate space
    view /= scale; // view is the view vector in texture coordinate space
    float sampleLen = 0.0;
    float delta = sampleSpacing*normalizedStep;
    vec4 sampleColor = vec4(0.0);
    float maxScalar = -100.0;
    while(true) {
        samplePos = rayStart + sampleLen * view;
        float scalar = texture3D(data, samplePos).s;
        if(maxScalar < scalar){
            maxScalar = scalar;
        }
        sampleLen += delta;
        if( samplePos.x > 1.01 || samplePos.y > 1.01 || samplePos.z > 1.01 ||
            samplePos.x < -0.01|| samplePos.y < -0.01|| samplePos.z < -0.01)
            break;
    }
    sampleColor = getColor(maxScalar,sampleSpacing);
    sampleColor.a = 0.5;
    gl_FragColor = sampleColor;
}
