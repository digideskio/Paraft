//varying vec3  textureCoords;
varying vec3 hitPoint;

void main()
{
        //gl_TexCoord[0].xyz = gl_Vertex.xyz;//(gl_TextureMatrix[0] * gl_MultiTexCoord0).xyz;
    gl_Position = ftransform();
    //gl_TexCoord[0] = gl_MultiTexCoord0;
    //textureCoords = gl_Vertex.xyz;
    hitPoint = gl_Vertex.xyz;
}
