import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
import math

GRAVITY = np.array([0, 0, -9.81])  # 假定重力方向为负z轴

class Cube:
    def __init__(self, edge_length, total_mass, spring_constant, offset=(0, 0, 0)):
        self.edge_length = edge_length
        self.mass_per_corner = total_mass / 12
        
        self.kinetic_energies = []
        self.potential_energies = []
        self.total_energies = []

        offset_z = 0.2
        self.masses = [
            self.Mass(self.mass_per_corner, [0, 0, 0+offset_z], 0),
            self.Mass(self.mass_per_corner, [0, 0, edge_length+offset_z], 1),
            self.Mass(self.mass_per_corner, [0, edge_length, 0+offset_z], 2),
            self.Mass(self.mass_per_corner, [0, edge_length, edge_length+offset_z], 3),
            self.Mass(self.mass_per_corner, [edge_length, 0, 0+offset_z], 4),
            self.Mass(self.mass_per_corner, [edge_length, 0, edge_length+offset_z], 5),
            self.Mass(self.mass_per_corner, [edge_length, edge_length, 0+offset_z], 6),
            self.Mass(self.mass_per_corner, [edge_length, edge_length, edge_length+offset_z], 7),
            self.Mass(self.mass_per_corner, [-edge_length, 0, 0+offset_z], 8),
            self.Mass(self.mass_per_corner, [-edge_length, 0, edge_length+offset_z], 9),
            self.Mass(self.mass_per_corner, [-edge_length, edge_length, 0+offset_z], 10),
            self.Mass(self.mass_per_corner, [-edge_length, edge_length, edge_length+offset_z], 11),
            self.Mass(self.mass_per_corner, [-2*edge_length, 0, 0+offset_z], 12),
            self.Mass(self.mass_per_corner, [-2*edge_length, 0, edge_length+offset_z], 13),
            self.Mass(self.mass_per_corner, [-2*edge_length, edge_length, 0+offset_z], 14),
            self.Mass(self.mass_per_corner, [-2*edge_length, edge_length, edge_length+offset_z], 15)
        ]
        
        self.springs = []
        for i in range(16):
            for j in range(i+1, 16):
                self.springs.append(self.Spring(spring_constant, self.masses[i], self.masses[j]))
        
        # for i in range(8,12):
        #     for j in range(i+1, 12):
        #         self.springs.append(self.Spring(spring_constant, self.masses[i], self.masses[j]))


    class Mass:
        def __init__(self, m, position,id):
            self.id = id
            self.m = m
            self.position = np.array(position, dtype=np.float64)
            self.velocity = np.array([0.0, 0.0, 0.0])
            self.acceleration = np.array([0.0, 0.0, 0.0])

        def apply_force(self, force):
            # F=ma => a = F/m
            self.acceleration = force / self.m

        def update_velocity(self, dt):
            # v = v + dt*a
            self.velocity += dt * self.acceleration

        def update_position(self, dt):
            # p = p + v*dt
            self.position += dt * self.velocity

        
        def apply_ground_collision(self, K_g=100000):
            if self.position[2] < 0:  # 质点位于地面以下
                force_ground = np.array([0, 0, -self.position[2] * K_g])
                self.apply_force(force_ground)

    class Spring:
        def __init__(self, k, m1, m2):
            self.k = k
            self.L0 = np.linalg.norm(m1.position - m2.position)
            self.a = self.L0
            self.m1 = m1
            self.m2 = m2
            self.b = 0.25
            self.c = math.pi

        def change_length(self, T):
        # 根据时间T改变弹簧的原始长度L0
            self.L0 = self.a + self.b * np.sin(2 * math.pi * T + self.c)

        def compute_force(self):
            # 计算弹簧的实际长度
            current_L = np.linalg.norm(self.m1.position - self.m2.position)
            
            # print('current_L',current_L)
            # 计算长度变化
            deltaL = current_L - 0.1
            
            # 力的大小: F=k*(L-L0)
            force_magnitude = self.k * deltaL
            
            # 力的方向: 从m1指向m2
            if current_L != 0:
                direction = (self.m2.position - self.m1.position) / current_L
            else:
                direction = np.array([0,0,0])
                
            force = force_magnitude * direction
            
            # # 计算摩擦力
            # relative_velocity = self.m1.velocity - self.m2.velocity
            # friction_force_magnitude = friction_coefficient * np.linalg.norm(relative_velocity)
            # if np.linalg.norm(relative_velocity) != 0:
            #     friction_direction = -relative_velocity / np.linalg.norm(relative_velocity)
            # else:
            #     friction_direction = np.array([0,0,0])
            
            # friction_force = friction_force_magnitude * friction_direction
            
            # # 综合考虑弹簧力和摩擦力
            # total_force = force + friction_force
            
            return force



def draw_cube(cube):
    glColor3f(0, 1, 0)  # 设置颜色为绿色
    glBegin(GL_LINES)
    for spring in cube.springs:
        glVertex3fv(spring.m1.position)
        glVertex3fv(spring.m2.position)
    glEnd()

    for mass in cube.masses:
        glPushMatrix()
        glTranslatef(*mass.position)
        glutSolidSphere(0.005, 10, 10)  # 小球来代表masses
        glPopMatrix()


def dynamic_simulation(cube, stop_criteria=None):
    T = 0  # 初始化时间
    DT = 0.005  # 时间步长
    k_ground = 100000  # 地面弹簧系数
    damping = 0.9999  # 阻尼系数
    # friction_mu_s = 1  # 静摩擦系数
    # friction_mu_k = 0.8  # 动摩擦系数

    count = 0
    while T < 0.5:
        # Time increment
        T += DT
        # Interaction step
        for spring in cube.springs:
        #     spring.change_length(T*10)
            # print('L0', spring.L0)
            if (spring.m1.id == 0 and spring.m2.id == 8) or (spring.m1.id == 8 and spring.m2.id == 0):
                spring.change_length(T*10)
            if (spring.m1.id == 2 and spring.m2.id == 10) or (spring.m1.id == 10 and spring.m2.id == 2):
                spring.change_length(T*10)
            if (spring.m1.id == 1 and spring.m2.id == 9) or (spring.m1.id == 9 and spring.m2.id == 1):
                spring.change_length(-T*10)
            if (spring.m1.id == 3 and spring.m2.id == 11) or (spring.m1.id == 11 and spring.m2.id == 3):
                spring.change_length(-T*10)

        for spring in cube.springs:
            force = spring.compute_force()
            # print('force', force)
            spring.m1.apply_force(force)
            spring.m2.apply_force(-force)  # 反向力

            # if spring.m1.id == 0 and spring.m2.id == 1:
                # print(f"Spring between mass {spring.m1.id} and {spring.m2.id}:")
                # print(f"Initial Length: {spring.L0}, Current Length: {np.linalg.norm(spring.m1.position - spring.m2.position)}")
                # print(f"Force applied on mass {spring.m1.id}: {force}")

        for mass in cube.masses:
            # if mass.id == 0 or mass.id == 1:
            #     print(f"\nMass {mass.id} before update:")
            #     print(f"Position: {mass.position}, Velocity: {mass.velocity}")
            mass.apply_ground_collision(k_ground)  # 添加地面碰撞反作用力
            mass.apply_force(GRAVITY * mass.m)  # 添加重力

            # 当与地面接触时，应用阻尼力和摩擦力
            if mass.position[2] <= 0:
                # 反向速度并应用阻尼
                if mass.velocity[2] < 0:
                    mass.velocity[2] = -damping * mass.velocity[2] 

                # 如果速度很小，则设为零
                # if abs(mass.velocity[2]) < 1e-3:
                #     mass.velocity[2] = 0

                # 计算水平摩擦力
                # friction = friction_mu_s if abs(mass.velocity[0]) < 1e-3 and abs(mass.velocity[1]) < 1e-3 else friction_mu_k
                # mass.velocity[0] *= 1 - friction
                # mass.velocity[1] *= 1 - friction

                # 如果位置穿越地面，则重新设定位置
                if mass.position[2] < 0:
                    mass.position[2] = 0

        # Integration Step
        for mass in cube.masses:
            mass.update_velocity(DT)
            mass.update_position(DT)
            mass.velocity[0] = mass.velocity[0] + 0.0001
            print('mass.velocity[0]', mass.velocity[0])

            # if mass.id == 8 or mass.id == 10:
            #     if count % 2 == 0:
            #         mass.position[0] = mass.position[0] +0.25
            #     else:
            #         mass.position[0] = mass.position[0] -0.25
            # if mass.id == 9 or mass.id == 11:
            #     if count % 2 == 0:
            #         mass.position[0] = mass.position[0] -0.25
            #     else:
            #         mass.position[0] = mass.position[0] +0.25
                # mass.position[0] = mass.position[0] * 0.25 * np.sin(2 * math.pi * T)

            # if mass.id == 0:
            #     kinetic_energies, potential_energies, total_energies = compute_energies(mass.m, mass.velocity[2], mass.position[2])
            #     cube.kinetic_energies.append(kinetic_energies)
            #     cube.potential_energies.append(potential_energies)
            #     cube.total_energies.append(total_energies)

                # print(f"\nMass {mass.id} after update:")
                # print(f"Position: {mass.position}, Velocity: {mass.velocity}")

        count += 1
        # 停止条件
        if stop_criteria and stop_criteria(cube):
            break

    return cube

def compute_energies(mass, velocities, heights, g=9.81):

    total_energies = [0.3 * mass * g]
    kinetic_energies = [0.5 * mass * velocities**2]
    potential_energies = [(0.3 * mass * g) - (0.5 * mass * velocities**2)]
    # potential_energies = [mass * g * heights]
    # total_energies = [kinetic_energies + potential_energies]
    
    return kinetic_energies, potential_energies, total_energies

def draw_ground():
    glColor3f(1, 1, 1)  # 设置颜色为白色
    gridSize = 10
    step = 0.5
    for x in np.arange(-gridSize, gridSize, step):
        glBegin(GL_LINES)
        glVertex3f(x, -gridSize, 0)
        glVertex3f(x, gridSize, 0)
        glEnd()
    for y in np.arange(-gridSize, gridSize, step):
        glBegin(GL_LINES)
        glVertex3f(-gridSize, y, 0)
        glVertex3f(gridSize, y, 0)
        glEnd()



# dynamic_simulation(test_cube) 

def main():
    pygame.init()
    glutInit()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -1)

    gluLookAt(-0.5, 0.3, 0.2,  # 相机的位置 (eyeX, eyeY, eyeZ)
          0, 0, 0.2,        # 相机的视点中心 (centerX, centerY, centerZ)
          0, 0, 1)        # 相机的向上方向 (upX, upY, upZ)
 
    test_cube = Cube(0.1, 0.8, 500000)

    for i in range(100):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        dynamic_simulation(test_cube)
    
        draw_cube(test_cube)
        draw_ground()
        pygame.display.flip()
        pygame.time.wait(10)

    # plt.figure(figsize=(10, 8))

    # # 第一个子图：动能
    # plt.subplot(2, 1, 1)  # 2行1列的布局，现在选择第一个子图
    # plt.plot(test_cube.kinetic_energies, label="Kinetic Energy")
    # plt.legend()
    # plt.ylabel("Energy (Joules)")
    # plt.title("Kinetic Energy vs Time")
    # plt.grid(True)

    # # 第二个子图：势能
    # plt.subplot(2, 1, 2)  # 2行1列的布局，现在选择第二个子图
    # plt.plot(test_cube.potential_energies, label="Potential Energy", color='green')
    # y_data = np.squeeze(test_cube.total_energies)
    # # y_data = np.squeeze(test_cube.total_energies)[:, 0]
    # plt.plot(y_data, label="Total Energy", color='red')
    # plt.legend()
    # plt.xlabel("Time (arbitrary units)")
    # plt.ylabel("Energy (Joules)")
    # plt.title("Potential & Total Energy vs Time")
    # plt.grid(True)

    # # 显示图形
    # plt.tight_layout()  # 调整子图的间距，使其不重叠
    # plt.show()

    # plt.plot(test_cube.kinetic_energies, label="Kinetic Energy")
    # plt.plot(test_cube.potential_energies, label="Potential Energy")
    # y_data = np.squeeze(test_cube.total_energies)[:, 0]
    # plt.plot(y_data, label="Total Energy")
    # plt.legend()
    # plt.xlabel("Time (arbitrary units)")
    # plt.ylabel("Energy (Joules)")
    # plt.title("Energy vs Time")
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
