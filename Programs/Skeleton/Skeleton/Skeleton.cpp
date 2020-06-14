//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tuska Jozsef Csongor
// Neptun : LAU37R
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================

#include "framework.h"

const int tessellationLevel = 40;

float rnd() { return (float)rand() / RAND_MAX; }

struct Material {

	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

struct Light {

	vec3 La, Le;
	vec4 wLightPos;

	vec4 qmul(vec4 q1, vec4 q2) {
		vec3 d1 = vec3(q1.x, q1.y, q1.z);
		vec3 d2 = vec3(q2.x, q2.y, q2.z);
		vec3 xyz = vec3(d2*q1.w + d1 * q2.w + cross(d1, d2));
		float w = q1.w*q2.w - dot(d1, d2);
		return vec4(xyz.x, xyz.y, xyz.z, w);
	}

	void Animate(float t) {

		vec4 q = vec4(sinf(t / 4.0f)*cosf(t) / 2.0f, sinf(t / 4.0f)*sinf(t) / 2.0f, sinf(t / 4.0f)*sqrtf(0.75f), cosf(t / 4.0f));
		vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, wLightPos), qinv);

		wLightPos = qr;
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

struct KleinTexture : public Texture {
	KleinTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(width * height);

		const vec3 green1(0.1f, 0.90f, 0.15f), green2(0.3f, 1.0f, 0.25f), green3(0.4f, 1.0f, 0.16f);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			float rand = rnd();

			if (rand < 0.5f)
				image[y * width + x] = green1;

			else if (0.5f <= rand && rand < 0.75f)
				image[y * width + x] = green2;

			else if (rand > 0.75f)
				image[y * width + x] = green3;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
};


struct DiniTexture : public Texture {
	DiniTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(width * height);

		const vec3 red(1.0, 0.10f, 0.2f), orange(1.0f, 0.64, 0);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			float rand = rnd();

			if (rand < 0.75f)
				image[y * width + x] = red;

			else
				image[y * width + x] = orange;

		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct RenderState {

	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

class Shader : public GPUProgram {

public:
	virtual void Bind(RenderState state) = 0;
};

class SymmetricShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv; 
		uniform Light[1] lights;    
		uniform int   nLights;
		uniform vec3  wEye;        
 
		layout(location = 0) in vec3  vtxPos;         
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal;
		out vec3 wView;             
		out vec3 wLight[1];		    
		out vec2 texcoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; 
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	const char * fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[1] lights;   
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal;      
		in  vec3 wView;        
		in  vec3 wLight[1];     
		in  vec2 texcoord;
		
        out vec4 fragmentColor;
 
		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess) / ((L+V)*(L+V))) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	SymmetricShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;
 
		uniform mat4  MVP, M, Minv; 
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         
 
		layout(location = 0) in vec3  vtxPos;          
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal, wView, wLight;				
		out vec2 texcoord;
 
		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; 
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	const char * fragmentSource = R"(
		#version 330
		precision highp float;
 
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal, wView, wLight;
		in  vec2 texcoord;
		out vec4 fragmentColor;    			
 
		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.lights[0].wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};


struct VertexData {

	vec3 position, normal;
	vec2 texcoord;
	vec3 drdU;
	vec3 drdV;
};

class Geometry {
protected:
	unsigned int vao;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

class Ellipsoid : public ParamSurface {
public:
	Ellipsoid() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;

		float U = u * 2.0f* (float)M_PI;
		float V = v * (float)M_PI;
		vd.position = vec3(cosf(U) * sinf(V), sinf(U) * sinf(V) / 2.0f, cosf(V) / 2.0f);
		vec3 drdU = vec3(-sinf(U)* sinf(V), cosf(U)*sinf(V) / 2.0f, 0);
		vec3 drdV = vec3(cosf(U)*cosf(V), sinf(U)*cosf(V) / 2.0f, -sinf(V) / 2.0f);
		vd.normal = cross(drdV, drdU);

		if (vd.position.z < 0) {
			vd.position.z = 0;
			vd.normal = vec3(0, 0, -1);
		}

		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Dini : public ParamSurface {
public:
	Dini() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;

		float U = u * 4.0f*(float)M_PI;
		float V = v + 0.01f;
		float a = 1, b = 0.15f;
		vd.position = vec3(a*cos(U)*sin(V), a * sinf(U) * sinf(V), a*(cosf(V) + log(tanf(V / 2)) + b * U));

		vec3 drdU = vec3(-sinf(U)*sinf(V)*a,
			cosf(U)*cosf(V)*a,
			b);

		vec3 drdV = vec3(a*cosf(U)*cosf(V),
			a*sinf(U)*cosf(V),
			1 / sinf(V) - sinf(V));
		vd.normal = cross(drdU, drdV);

		vd.texcoord = vec2(u, v);
		return vd;
	}
};


class Klein : public ParamSurface {
public:
	Klein() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float fx, fy, fz;

		float U = u * 2.0f * (float)M_PI, V = v * 2.0f * (float)M_PI;
		float a = 6.0f * cosf(U) * (1 + sinf(U));
		float b = 16.0f * sinf(U);
		float c = 4.0f * (1.0f - cosf(U) / 2.0f);

		if ((float)M_PI < U  && U <= (float)M_PI * 2.0f) {
			fx = a + c * cosf(V + (float)M_PI);
			fy = b;
			fz = c * sinf(V);

			vec3 drdU = vec3(6.0f * cosf(U)*cosf(U) - 2.0f*sinf(U)*(3.0f*sinf(U) + cosf(V) + 3.0f),
				16.0f*cosf(U),
				2.0f*sinf(U)*sinf(V));

			vec3 drdV = vec3(-2 * (cosf(U) - 2.0f)*sinf(V),
				0,
				-2.0f*(cosf(U) - 2)*cosf(V));

			vd.normal = cross(drdU, drdV);
			vd.position = vec3(fx, fy, fz);
			vd.drdU = drdU;
			vd.drdV = drdV;
		}

		else {
			fx = a + c * cosf(U) * cosf(V);
			fy = b + c * sinf(U) * cosf(V);
			fz = c * sinf(V);

			vec3 drdU = vec3(4 * sinf(U)*cosf(U)*cosf(V) - 2 * sinf(U)*(3 * sinf(U) + 2 * cosf(V) + 3) + 6 * cosf(U)*cosf(U),
				-2 * cosf(U)*cosf(U)*cosf(V) + 4 * cosf(U)*(cosf(V) + 4.0f) + 2.0f*sinf(U)*sinf(U)*cosf(V),
				2.0f * sinf(U)*sinf(V));

			vec3 drdV = vec3(2 * (cosf(U) - 2)*cosf(U)*sinf(V),
				2.0f*sinf(U)*(cosf(U) - 2)*sinf(V),
				4.0f * (1.0f - cosf(U) / 2.0f)*cosf(V));

			vd.normal = cross(drdU, drdV);
			vd.drdU = drdU;
			vd.drdV = drdV;
			vd.position = vec3(fx, fy, fz);
		}

		vd.texcoord = vec2(u, v);

		return vd;
	}
};

bool zoom = false;
struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 80;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() {
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(vec2 pos) {

		ParamSurface * kleinSurface = new Klein();

		VertexData tr = kleinSurface->GenVertexData(pos.x, pos.y);

		vec3 i = normalize(tr.drdU + tr.drdV);
		vec3 k = normalize(cross(tr.drdU, tr.drdV));
		vec3 j = normalize(cross(i, k));

		wEye = tr.position + k * 2;
		wLookat = tr.position + i;
		wVup = k;

		if (zoom) {
			wEye = tr.position + k * 2;
			wLookat = tr.position + i;
			wVup = k;
			fov = 75.0f * (float)M_PI / 180.0f;
		}

		else {
			fov = 75.0f * (float)M_PI / 180.0f;
			wEye = tr.position + k * 20;
			wLookat = tr.position + i;
			wVup = k;
		}

		delete kleinSurface;
	}
};

struct KaticaTexture : public Texture {
	KaticaTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(width * height);

		ParamSurface * surface = new Ellipsoid();
		std::vector<vec2> dots;
		dots.reserve(7);

		dots.push_back(vec2(1.0f, 0.2f));
		dots.push_back(vec2(0.8f, 0.35f));
		dots.push_back(vec2(0.7f, 0.35f));
		dots.push_back(vec2(0.75f, 0.15f));
		dots.push_back(vec2(0.2f, 0.35f));
		dots.push_back(vec2(0.3f, 0.35f));
		dots.push_back(vec2(0.25f, 0.15f));

		const vec3 red(1, 0, 0), black(0, 0, 0);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++) {
				{
					float u = (float)x / width;
					float v = (float)y / height;

					image[y * width + x] = red;

					for (int i = 0; i < 7; i++) {
						if (length(surface->GenVertexData(u, v).position - surface->GenVertexData(dots[i].x, dots[i].y).position) < 0.1f) {
							image[y * width + x] = black;
						}
					}
				}
			}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		delete surface;
	}
};

struct Object {
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {}
};

struct Kitty : public Object {
	vec2 pos;
	float angle = 0.0f;
public:
	Kitty(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		Object(_shader, _material, _texture, _geometry) {

		ParamSurface * kleinSurface = new Klein();
		pos = vec2(0.4f, 0.4f);
		delete kleinSurface;
	}

	void Draw(RenderState state) {

		ParamSurface * kleinSurface = new Klein();
		VertexData vd = kleinSurface->GenVertexData(pos.x, pos.y);

		vec3 i = normalize(vd.drdU + vd.drdV);
		vec3 k = normalize(vd.normal);
		vec3 j = normalize(cross(k, i));

		mat4 RotateTranslate = mat4(i.x, i.y, i.z, 0,
									j.x, j.y, j.z, 0,
									k.x, k.y, k.z, 0,
			vd.position.x, vd.position.y, vd.position.z, 1);

		mat4 RotateTranslateInv = mat4(i.x, j.x, k.x, 0,
										i.y, j.y, k.y, 0,
										i.z, j.z, k.z, 0,
									-vd.position.x, -vd.position.y, -vd.position.z, 1);

		state.M = ScaleMatrix(scale) * RotateTranslate;
		state.Minv = RotateTranslateInv * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
		delete kleinSurface;
	}

	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		ParamSurface * kleinSurface = new Klein();
		VertexData sf = kleinSurface->GenVertexData(pos.x, pos.y);
		float du = 1.0f * dt*cosf(angle) / length(sf.drdU);
		float dv = 1.0f * dt*sinf(angle) / length(sf.drdV);
		pos.x += du;
		pos.y += dv;
		delete kleinSurface;
	}
};

class Scene {
	std::vector<Object *> objects;
	Kitty * kitty;
public:
	Camera camera;
	std::vector<Light> lights;

	void Build() {

		Shader * symmetricShader = new SymmetricShader();
		Shader * nprShader = new NPRShader();

		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material * material1 = new Material;
		material1->kd = vec3(0.8, 0.6, 0.4);
		material1->ks = vec3(0.3, 0.3, 0.3);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		Texture * diniTexture = new DiniTexture(200, 400);
		Texture * katicaTexture = new KaticaTexture(1000, 1000);
		Texture * kleinTexture = new KleinTexture(300, 600);

		Geometry * sphere = new Ellipsoid();
		Geometry * klein = new Klein();
		Geometry * dini = new Dini();

		ParamSurface * kleinSurface = new Klein();

		kitty = new Kitty(nprShader, material0, katicaTexture, sphere);
		kitty->scale = vec3(1, 1, 1);
		kitty->rotationAngle = 0;
		objects.push_back(kitty);

		Object * kleinObject = new Object(symmetricShader, material0, kleinTexture, klein);
		kleinObject->translation = vec3(0, 0, 0);
		kleinObject->rotationAxis = vec3(0, 1, 0);
		kleinObject->scale = vec3(1.0f, 1.0f, 1.0f);
		objects.push_back(kleinObject);

		VertexData tr = kleinSurface->GenVertexData(0.4f, 0.4f);

		float cosf = dot(vec3(0, 0, 1), normalize(tr.normal));
		float angle = acosf(cosf);

		Object * diniObject1 = new Object(symmetricShader, material0, diniTexture, dini);
		diniObject1->translation = tr.position*kleinObject->scale + normalize(tr.normal)*2.0f;
		diniObject1->rotationAxis = cross(vec3(0, 0, 1), normalize(tr.normal));
		diniObject1->rotationAngle = angle;
		objects.push_back(diniObject1);

		VertexData tr2 = kleinSurface->GenVertexData(0.4, 0.2f);

		float cosf2 = dot(vec3(0, 0, 1), normalize(tr2.normal));
		float angle2 = acosf(cosf2);

		Object * diniObject2 = new Object(symmetricShader, material0, diniTexture, dini);
		diniObject2->translation = tr2.position + normalize(tr2.normal)*2.0f;
		diniObject2->rotationAxis = cross(vec3(0, 0, 1), normalize(tr2.normal));
		diniObject2->rotationAngle = angle2;
		objects.push_back(diniObject2);

		camera.wEye = vec3(0, 0, 30);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(1);
		lights[0].wLightPos = vec4(0, 0, 1, 0);
		lights[0].La = vec3(0.1, 0.1, 1);
		lights[0].Le = vec3(3, 3, 3);

		delete kleinSurface;

	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		lights[0].Animate(tend - tstart);
		for (Object * obj : objects) obj->Animate(tstart, tend);
		camera.Animate(kitty->pos);
	}

	void SetAngle(int num) {
		kitty->angle += num * (float)M_PI / 4.0f;
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {


	if (key == 'a')
		scene.SetAngle(1);

	if (key == 'd')
		scene.SetAngle(-1);

	if (key == ' ')
		zoom = !zoom;
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}